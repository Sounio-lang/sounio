//! Salsa-inspired query system for incremental computation

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Revision number for tracking freshness
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Revision(pub u64);

impl Revision {
    pub fn next(&self) -> Self {
        Revision(self.0 + 1)
    }
}

/// Durability level for inputs
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Durability {
    /// Changes frequently (source text)
    Low,
    /// Changes occasionally (dependencies)
    Medium,
    /// Rarely changes (configuration)
    High,
}

impl Default for Durability {
    fn default() -> Self {
        Durability::Low
    }
}

/// Trait for query keys
pub trait QueryKey: Clone + Hash + Eq + Send + Sync + 'static {
    type Value: Clone + Send + Sync + 'static;
}

/// A stored query result
#[derive(Debug)]
struct QuerySlot<K: QueryKey> {
    /// The cached value
    value: Option<K::Value>,

    /// Revision when this was computed
    computed_at: Revision,

    /// Revision when this was last verified
    verified_at: Revision,

    /// Dependencies of this computation
    dependencies: Vec<DependencyKey>,

    /// Durability of the input
    durability: Durability,
}

/// A key for tracking dependencies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DependencyKey {
    type_id: TypeId,
    key_hash: u64,
}

impl DependencyKey {
    fn new<K: QueryKey>(key: &K) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);

        DependencyKey {
            type_id: TypeId::of::<K>(),
            key_hash: hasher.finish(),
        }
    }
}

/// Storage for a specific query type
struct QueryStorage<K: QueryKey> {
    slots: RwLock<HashMap<K, QuerySlot<K>>>,
}

impl<K: QueryKey> QueryStorage<K> {
    fn new() -> Self {
        QueryStorage {
            slots: RwLock::new(HashMap::new()),
        }
    }

    fn get(&self, key: &K) -> Option<(K::Value, Revision)> {
        let slots = self.slots.read().unwrap();
        slots
            .get(key)
            .and_then(|slot| slot.value.clone().map(|v| (v, slot.computed_at)))
    }

    fn set(
        &self,
        key: K,
        value: K::Value,
        computed_at: Revision,
        dependencies: Vec<DependencyKey>,
        durability: Durability,
    ) {
        let mut slots = self.slots.write().unwrap();
        slots.insert(
            key,
            QuerySlot {
                value: Some(value),
                computed_at,
                verified_at: computed_at,
                dependencies,
                durability,
            },
        );
    }

    fn invalidate(&self, key: &K) {
        let mut slots = self.slots.write().unwrap();
        slots.remove(key);
    }

    fn invalidate_all(&self) {
        let mut slots = self.slots.write().unwrap();
        slots.clear();
    }
}

/// Trait for type-erased query storage
trait AnyQueryStorage: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn invalidate_all(&self);
}

impl<K: QueryKey> AnyQueryStorage for QueryStorage<K> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn invalidate_all(&self) {
        self.invalidate_all();
    }
}

/// The main query database
pub struct QueryDatabase {
    /// Current revision
    current_revision: RwLock<Revision>,

    /// Revision for each durability level
    durability_revisions: RwLock<HashMap<Durability, Revision>>,

    /// Storage for each query type
    storage: RwLock<HashMap<TypeId, Arc<dyn AnyQueryStorage>>>,

    /// Dependency tracker for current computation
    dependency_tracker: RwLock<Vec<DependencyKey>>,
}

impl QueryDatabase {
    pub fn new() -> Self {
        QueryDatabase {
            current_revision: RwLock::new(Revision(0)),
            durability_revisions: RwLock::new(HashMap::new()),
            storage: RwLock::new(HashMap::new()),
            dependency_tracker: RwLock::new(Vec::new()),
        }
    }

    /// Get the current revision
    pub fn current_revision(&self) -> Revision {
        *self.current_revision.read().unwrap()
    }

    /// Increment the revision (called when inputs change)
    pub fn new_revision(&self, durability: Durability) -> Revision {
        let mut rev = self.current_revision.write().unwrap();
        *rev = rev.next();

        let mut dur_revs = self.durability_revisions.write().unwrap();
        dur_revs.insert(durability, *rev);

        *rev
    }

    /// Get or compute a query
    pub fn query<K: QueryKey>(
        &self,
        key: K,
        compute: impl FnOnce(&Self, &K) -> K::Value,
    ) -> K::Value {
        // Check cache
        let storage = self.get_or_create_storage::<K>();

        if let Some((value, computed_at)) = storage.get(&key) {
            // Verify the value is still valid
            if self.is_valid::<K>(&key, computed_at) {
                // Track dependency
                self.track_dependency::<K>(&key);
                return value;
            }
        }

        // Compute the value
        let deps_before = self.start_tracking();
        let value = compute(self, &key);
        let dependencies = self.end_tracking(deps_before);

        // Cache the result
        let computed_at = self.current_revision();
        storage.set(
            key.clone(),
            value.clone(),
            computed_at,
            dependencies,
            Durability::Low,
        );

        // Track dependency
        self.track_dependency::<K>(&key);

        value
    }

    /// Set an input value
    pub fn set_input<K: QueryKey>(&self, key: K, value: K::Value, durability: Durability) {
        let storage = self.get_or_create_storage::<K>();
        let revision = self.new_revision(durability);
        storage.set(key, value, revision, Vec::new(), durability);
    }

    /// Invalidate a specific query
    pub fn invalidate<K: QueryKey>(&self, key: &K) {
        if let Some(storage) = self.get_storage::<K>() {
            storage.invalidate(key);
        }
    }

    fn get_or_create_storage<K: QueryKey>(&self) -> Arc<QueryStorage<K>> {
        let type_id = TypeId::of::<K>();

        // Try to get existing storage
        {
            let storage = self.storage.read().unwrap();
            if let Some(s) = storage.get(&type_id) {
                return s
                    .as_any()
                    .downcast_ref::<QueryStorage<K>>()
                    .map(|_| Arc::clone(s) as Arc<QueryStorage<K>>)
                    .unwrap_or_else(|| Arc::new(QueryStorage::new()));
            }
        }

        // Create new storage
        let new_storage = Arc::new(QueryStorage::<K>::new());
        {
            let mut storage = self.storage.write().unwrap();
            storage.insert(
                type_id,
                Arc::clone(&new_storage) as Arc<dyn AnyQueryStorage>,
            );
        }

        new_storage
    }

    fn get_storage<K: QueryKey>(&self) -> Option<Arc<QueryStorage<K>>> {
        let storage = self.storage.read().unwrap();
        storage
            .get(&TypeId::of::<K>())
            .map(|s| Arc::clone(s) as Arc<QueryStorage<K>>)
    }

    fn is_valid<K: QueryKey>(&self, _key: &K, computed_at: Revision) -> bool {
        computed_at >= self.current_revision()
    }

    fn track_dependency<K: QueryKey>(&self, key: &K) {
        let dep = DependencyKey::new(key);
        let mut tracker = self.dependency_tracker.write().unwrap();
        if !tracker.contains(&dep) {
            tracker.push(dep);
        }
    }

    fn start_tracking(&self) -> Vec<DependencyKey> {
        let mut tracker = self.dependency_tracker.write().unwrap();
        std::mem::take(&mut *tracker)
    }

    fn end_tracking(&self, previous: Vec<DependencyKey>) -> Vec<DependencyKey> {
        let mut tracker = self.dependency_tracker.write().unwrap();
        let current = std::mem::replace(&mut *tracker, previous);
        current
    }
}

impl Default for QueryDatabase {
    fn default() -> Self {
        Self::new()
    }
}
