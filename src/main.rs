use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    sync::{Arc, Mutex},
};
use hnsw_rs::prelude::*;
use dotenvy::dotenv;

#[derive(Clone, Serialize, Deserialize)]
struct CollectionConfig {
    distance: String, // "l2" or "cosine"
    hnsw: HnswParams,
}

#[derive(Clone, Serialize, Deserialize)]
struct HnswParams {
    max_nb_connection: usize,
    ef_search: usize,
    max_elements: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct VectorRecord {
    id: u64,
    vector: Vec<f32>,
    payload: serde_json::Value,
}

struct Collection<'a> {
    config: CollectionConfig,
    records: Vec<VectorRecord>,
    hnsw_l2: Option<Arc<Hnsw<'a, f32, DistL2>>>,
    hnsw_cosine: Option<Arc<Hnsw<'a, f32, DistCosine>>>,
}

impl<'a> Collection<'a> {
    fn new(config: CollectionConfig, dim: usize) -> Self {
        let hnsw_l2 = if config.distance == "l2" {
            Some(Arc::new(Hnsw::new(
                config.hnsw.max_nb_connection,
                16,                // efConstruction
                dim,               // vector dimension
                config.hnsw.max_elements,
                DistL2 {},
            )))
        } else {
            None
        };

        let hnsw_cosine = if config.distance == "cosine" {
            Some(Arc::new(Hnsw::new(
                config.hnsw.max_nb_connection,
                16,                // efConstruction
                dim,               // vector dimension
                config.hnsw.max_elements,
                DistCosine {},
            )))
        } else {
            None
        };

        Self {
            config,
            records: Vec::new(),
            hnsw_l2,
            hnsw_cosine,
        }
    }

    fn upsert(&mut self, ids: Vec<u64>, vectors: Vec<Vec<f32>>, payloads: Vec<serde_json::Value>) {
        for (i, id) in ids.iter().enumerate() {
            let record = VectorRecord {
                id: *id,
                vector: vectors[i].clone(),
                payload: payloads[i].clone(),
            };
            if let Some(hnsw) = &self.hnsw_l2 {
                hnsw.insert((vectors[i].as_slice(), *id as usize));
            }
            if let Some(hnsw) = &self.hnsw_cosine {
                hnsw.insert((vectors[i].as_slice(), *id as usize));
            }
            self.records.push(record);
        }
    }

    fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(u64, f32)> {
        if let Some(hnsw) = &self.hnsw_l2 {
            let res = hnsw.search(query.as_slice(), top_k, self.config.hnsw.ef_search);
            return res.into_iter().map(|n| (n.d_id as u64, n.distance)).collect();
        }
        if let Some(hnsw) = &self.hnsw_cosine {
            let res = hnsw.search(query.as_slice(), top_k, self.config.hnsw.ef_search);
            return res.into_iter().map(|n| (n.d_id as u64, n.distance)).collect();
        }
        vec![]
    }
}

struct AppState<'a> {
    collections: Mutex<HashMap<String, Collection<'a>>>,
}

#[derive(Deserialize)]
struct CreateCollectionBody {
    name: String,
    config: CollectionConfig,
    dim: usize,
}

async fn create_collection<'a>(
    data: web::Data<AppState<'a>>,
    body: web::Json<CreateCollectionBody>,
) -> impl Responder {
    let mut collections = data.collections.lock().unwrap();
    collections.insert(
        body.name.clone(),
        Collection::new(body.config.clone(), body.dim),
    );
    HttpResponse::Ok().finish()
}

#[derive(Deserialize)]
struct UpsertBody {
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    payloads: Vec<serde_json::Value>,
}

async fn upsert_vectors<'a>(
    data: web::Data<AppState<'a>>,
    path: web::Path<String>,
    body: web::Json<UpsertBody>,
) -> impl Responder {
    let mut collections = data.collections.lock().unwrap();
    if let Some(coll) = collections.get_mut(&path.into_inner()) {
        coll.upsert(body.ids.clone(), body.vectors.clone(), body.payloads.clone());
        HttpResponse::Ok().finish()
    } else {
        HttpResponse::NotFound().body("Collection not found")
    }
}

#[derive(Deserialize)]
struct SearchBody {
    query: Vec<f32>,
    top_k: usize,
}

async fn search_vectors<'a>(
    data: web::Data<AppState<'a>>,
    path: web::Path<String>,
    body: web::Json<SearchBody>,
) -> impl Responder {
    let collections = data.collections.lock().unwrap();
    if let Some(coll) = collections.get(&path.into_inner()) {
        let results = coll.search(body.query.clone(), body.top_k);
        HttpResponse::Ok().json(results)
    } else {
        HttpResponse::NotFound().body("Collection not found")
    }
}

async fn list_collections<'a>(data: web::Data<AppState<'a>>) -> impl Responder {
    let collections = data.collections.lock().unwrap();
    let names: Vec<String> = collections.keys().cloned().collect();
    HttpResponse::Ok().json(names)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    let port: u16 = env::var("PORT").unwrap_or_else(|_| "5202".to_string()).parse().unwrap();

    let state = web::Data::new(AppState {
        collections: Mutex::new(HashMap::new()),
    });

    println!("Server running on 127.0.0.1:{}", port);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/collections", web::get().to(list_collections))
            .route("/collections", web::post().to(create_collection))
            .route("/collections/{name}/upsert", web::post().to(upsert_vectors))
            .route("/collections/{name}/search", web::post().to(search_vectors))
    })
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
