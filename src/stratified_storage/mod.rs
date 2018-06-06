mod strata;
mod mpmc_map;
mod assigners;
mod selectors;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use commons::ExampleWithScore;


type WeightsTable = Arc<RwLock<HashMap<i8, f64>>>;