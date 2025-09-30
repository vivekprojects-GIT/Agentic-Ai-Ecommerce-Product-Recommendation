"""ChromaDB-based product catalog with semantic search."""

import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)

class CatalogStore:
    """Product catalog using ChromaDB for vector storage and semantic search."""
    
    def __init__(self, persist_dir: str = "./chroma_db", embedding_model: str = None):
        # Resolve configuration from environment with sensible defaults
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", persist_dir)
        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en")

        # Prefer CPU to reduce RAM/GPU pressure
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model, device="cpu")
        logger.info(f"Loaded embedding model: {embedding_model}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="products",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load catalog data
        self._load_catalog()
        
        logger.info("Catalog initialized with 20 products")
    
    def _load_catalog(self):
        """Load product catalog from JSON file."""
        try:
            catalog_path = "src/data/catalog.json"
            
            # If collection already has data and not forcing reload, skip loading
            try:
                existing_count = self.collection.count()
            except Exception:
                existing_count = 0

            force_reload = os.getenv("CHROMA_FORCE_RELOAD", "0").lower() in ["1", "true", "yes"]
            if existing_count > 0 and not force_reload:
                logger.info(f"Chroma collection already has {existing_count} items. Skipping catalog reload.")
                return

            # Load catalog data
            with open(catalog_path, 'r') as f:
                products = json.load(f)
            
            logger.info(f"Loading {len(products)} products from catalog")
            
            try:
                # Safer reset: delete collection then recreate to avoid filter issues
                name = self.collection.name
                self.client.delete_collection(name)
                self.collection = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                # Fallback: ignore if cannot delete
                pass
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for product in products:
                # Use search_text as document
                documents.append(product['search_text'])
                
                # Create metadata (all other fields)
                metadata = self._create_metadata(product)
                metadatas.append(metadata)
                
                # Use product ID as string (Chroma expects string IDs)
                try:
                    ids.append(str(product['id']))
                except Exception:
                    ids.append(str(product.get('id', '')))  # ensure string
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info("Successfully loaded products with embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            raise e
    
    def _create_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata dictionary for ChromaDB.
        Adds stable fields including a product URL. If no URL provided in the
        source JSON, constructs one from PRODUCT_BASE_URL and product id.
        """
        base_url = os.getenv("PRODUCT_BASE_URL", "https://example.com/products/")
        product_url = product.get("url") or f"{base_url}{product['id']}"

        metadata = {
            'id': product['id'],
            'name': product['name'],
            'description': product['description'],
            'price': product['price'],
            'availability': product['availability'],
            'brand': product['attributes'].get('brand', ''),
            'color_family': product['attributes'].get('color_family', ''),
            'material': product['attributes'].get('material', ''),
            'category': product['category'][0] if product['category'] else '',  # Use first category
            'size': json.dumps(product['attributes'].get('size', [])),  # Store as JSON string
            'url': product_url
        }
        return metadata
    
    def _metadata_to_product(self, metadata: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Convert ChromaDB metadata back to product dictionary."""
        product = {
            'id': metadata.get('id', ''),
            'name': metadata.get('name', ''),
            'description': metadata.get('description', ''),
            'price': metadata.get('price', 0),
            'availability': metadata.get('availability', True),
            'category': [metadata.get('category', '')] if metadata.get('category') else [],
            'attributes': {
                'brand': metadata.get('brand', ''),
                'color_family': metadata.get('color_family', ''),
                'material': metadata.get('material', ''),
                'size': json.loads(metadata.get('size', '[]'))
            },
            'search_text': document,
            'url': metadata.get('url', '')
        }
        return product
    
    def search(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search products using semantic similarity with optional metadata filtering."""
        try:
            # Build where clause for metadata filtering
            where_clause = self._build_filters(filters) if filters else None
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            # Convert results to product format
            products = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    product = self._metadata_to_product(metadata, doc)
                    products.append(product)
            
            logger.info(f"Found {len(products)} products for query: '{query}'")
            return products
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products from the catalog for hybrid search."""
        try:
            # Get all products from ChromaDB
            results = self.collection.get()
            
            products = []
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    document = results['documents'][i] if results['documents'] else ""
                    product = self._metadata_to_product(metadata, document)
                    products.append(product)
            
            logger.info(f"Retrieved {len(products)} products for hybrid search")
            return products
            
        except Exception as e:
            logger.error(f"Failed to get all products: {e}")
            return []
    
    def _build_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause for metadata filtering."""
        if not filters:
            return None
            
        # For ChromaDB, we need to use $and for multiple conditions
        conditions = []
        
        # Price filtering
        if 'price_max' in filters and 'price_min' in filters:
            conditions.append({
                '$and': [
                    {'price': {'$gte': filters['price_min']}},
                    {'price': {'$lte': filters['price_max']}}
                ]
            })
        elif 'price_max' in filters:
            conditions.append({'price': {'$lte': filters['price_max']}})
        elif 'price_min' in filters:
            conditions.append({'price': {'$gte': filters['price_min']}})
        
        # Color filtering
        if 'color_family' in filters:
            conditions.append({'color_family': {'$eq': filters['color_family']}})
        
        # Brand filtering
        if 'brand' in filters:
            conditions.append({'brand': {'$eq': filters['brand']}})
        
        # Category filtering
        if 'category' in filters:
            conditions.append({'category': {'$eq': filters['category']}})
        
        # Availability filtering
        if 'availability' in filters:
            conditions.append({'availability': {'$eq': filters['availability']}})
        
        # Combine all conditions
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {'$and': conditions}
        else:
            return None
