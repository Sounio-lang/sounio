//! Cache-Line Packing: Optimizing struct layout for cache efficiency.
//!
//! This module reorders struct fields to minimize cache line splits
//! and maximize co-access within cache lines.

use super::access::AccessPattern;
use std::collections::{HashMap, HashSet};

/// A field in a struct being packed.
#[derive(Debug, Clone)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Size in bytes
    pub size: usize,
    /// Alignment requirement
    pub alignment: usize,
    /// Original position in struct
    pub original_position: usize,
    /// Semantic type (for semantic-aware packing)
    pub semantic_type: Option<String>,
}

impl Field {
    /// Create a new field.
    pub fn new(name: impl Into<String>, size: usize, alignment: usize) -> Self {
        Self {
            name: name.into(),
            size,
            alignment,
            original_position: 0,
            semantic_type: None,
        }
    }

    /// Set the original position.
    pub fn at_position(mut self, pos: usize) -> Self {
        self.original_position = pos;
        self
    }

    /// Set the semantic type.
    pub fn with_semantic_type(mut self, ty: impl Into<String>) -> Self {
        self.semantic_type = Some(ty.into());
        self
    }
}

/// A group of fields that should be packed together.
#[derive(Debug, Clone)]
pub struct FieldGroup {
    /// Fields in this group
    pub fields: Vec<String>,
    /// Total size of the group
    pub total_size: usize,
    /// Co-access correlation (0.0 to 1.0)
    pub correlation: f64,
    /// Group hotness
    pub is_hot: bool,
}

impl FieldGroup {
    /// Create a new field group.
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            total_size: 0,
            correlation: 0.0,
            is_hot: false,
        }
    }

    /// Add a field to the group.
    pub fn add(&mut self, name: impl Into<String>, size: usize) {
        self.fields.push(name.into());
        self.total_size += size;
    }

    /// Check if the group fits in a cache line.
    pub fn fits_in_cache_line(&self, cache_line_size: usize) -> bool {
        self.total_size <= cache_line_size
    }
}

impl Default for FieldGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// A packed struct layout.
#[derive(Debug, Clone)]
pub struct PackedLayout {
    /// The struct name
    pub struct_name: String,
    /// Fields in their new order
    pub fields: Vec<PackedField>,
    /// Total size after packing
    pub total_size: usize,
    /// Padding bytes
    pub padding: usize,
    /// Number of cache lines needed
    pub cache_lines: usize,
    /// Field groups
    pub groups: Vec<FieldGroup>,
    /// Improvement over original (0.0 = no change, 1.0 = perfect)
    pub improvement: f64,
}

/// A field in the packed layout.
#[derive(Debug, Clone)]
pub struct PackedField {
    /// Field name
    pub name: String,
    /// Offset in bytes from struct start
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
    /// Which cache line this field starts in
    pub cache_line: usize,
    /// Whether this field spans multiple cache lines
    pub spans_cache_line: bool,
    /// Original position (for comparison)
    pub original_position: usize,
}

impl PackedLayout {
    /// Create a new packed layout.
    pub fn new(struct_name: impl Into<String>) -> Self {
        Self {
            struct_name: struct_name.into(),
            fields: Vec::new(),
            total_size: 0,
            padding: 0,
            cache_lines: 0,
            groups: Vec::new(),
            improvement: 0.0,
        }
    }

    /// Get a field by name.
    pub fn get_field(&self, name: &str) -> Option<&PackedField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Get the new offset for a field.
    pub fn offset_of(&self, name: &str) -> Option<usize> {
        self.get_field(name).map(|f| f.offset)
    }

    /// Check if two fields are in the same cache line.
    pub fn same_cache_line(&self, field1: &str, field2: &str) -> bool {
        match (self.get_field(field1), self.get_field(field2)) {
            (Some(f1), Some(f2)) => f1.cache_line == f2.cache_line,
            _ => false,
        }
    }

    /// Generate code comments showing the layout.
    pub fn to_comments(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("// Packed layout for {}\n", self.struct_name));
        out.push_str(&format!(
            "// Total size: {} bytes ({} cache lines)\n",
            self.total_size, self.cache_lines
        ));
        out.push_str(&format!("// Padding: {} bytes\n", self.padding));
        out.push_str(&format!(
            "// Improvement: {:.1}%\n",
            self.improvement * 100.0
        ));
        out.push_str("//\n");
        out.push_str("// Offset  Size  CL  Field\n");
        out.push_str("// ------  ----  --  -----\n");

        for field in &self.fields {
            let span = if field.spans_cache_line { "*" } else { " " };
            out.push_str(&format!(
                "// {:>6}  {:>4}  {:>2}{}  {}\n",
                field.offset, field.size, field.cache_line, span, field.name
            ));
        }

        out
    }

    /// Generate the new struct definition.
    pub fn to_struct_def(&self, original_types: &HashMap<String, String>) -> String {
        let mut out = String::new();
        out.push_str(&format!("#[repr(C)]\nstruct {} {{\n", self.struct_name));

        for field in &self.fields {
            let ty = original_types
                .get(&field.name)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            out.push_str(&format!("    {}: {},\n", field.name, ty));
        }

        out.push_str("}\n");
        out
    }
}

/// The cache-line packer.
pub struct CacheLinePacker {
    /// Cache line size in bytes
    cache_line_size: usize,
    /// Minimum alignment
    min_alignment: usize,
    /// Whether to prefer hot fields first
    hot_first: bool,
    /// Whether to respect semantic grouping
    semantic_aware: bool,
}

impl CacheLinePacker {
    /// Create a new packer with default settings.
    pub fn new(cache_line_size: usize) -> Self {
        Self {
            cache_line_size,
            min_alignment: 1,
            hot_first: true,
            semantic_aware: true,
        }
    }

    /// Set minimum alignment.
    pub fn with_min_alignment(mut self, alignment: usize) -> Self {
        self.min_alignment = alignment;
        self
    }

    /// Disable hot-first ordering.
    pub fn without_hot_first(mut self) -> Self {
        self.hot_first = false;
        self
    }

    /// Disable semantic awareness.
    pub fn without_semantic_awareness(mut self) -> Self {
        self.semantic_aware = false;
        self
    }

    /// Pack fields based on access patterns.
    pub fn pack(
        &self,
        fields: &[(&str, usize, &str)], // (name, size, semantic_type)
        patterns: &HashMap<String, AccessPattern>,
    ) -> PackedLayout {
        // Build field list
        let field_list: Vec<Field> = fields
            .iter()
            .enumerate()
            .map(|(i, (name, size, sem_type))| {
                let alignment = self.compute_alignment(*size);
                Field::new(*name, *size, alignment)
                    .at_position(i)
                    .with_semantic_type(*sem_type)
            })
            .collect();

        // Extract co-access information
        let co_accesses = self.extract_co_accesses(&field_list, patterns);

        // Build field groups
        let groups = self.build_groups(&field_list, &co_accesses);

        // Determine field ordering
        let ordering = self.compute_ordering(&field_list, &groups, patterns);

        // Place fields and compute layout
        self.compute_layout(&field_list, &ordering, &groups)
    }

    /// Pack with simple co-access pairs.
    pub fn pack_simple(
        &self,
        fields: &[(&str, usize)],
        co_accesses: &[((&str, &str), f64)],
    ) -> PackedLayout {
        let field_list: Vec<Field> = fields
            .iter()
            .enumerate()
            .map(|(i, (name, size))| {
                let alignment = self.compute_alignment(*size);
                Field::new(*name, *size, alignment).at_position(i)
            })
            .collect();

        let co_access_map: Vec<(String, String, f64)> = co_accesses
            .iter()
            .map(|((a, b), corr)| (a.to_string(), b.to_string(), *corr))
            .collect();

        let groups = self.build_groups_simple(&field_list, &co_access_map);
        let ordering = self.compute_simple_ordering(&field_list, &groups);

        self.compute_layout(&field_list, &ordering, &groups)
    }

    /// Compute alignment for a field size.
    fn compute_alignment(&self, size: usize) -> usize {
        match size {
            1 => 1,
            2 => 2,
            3..=4 => 4,
            5..=8 => 8,
            _ => 8.max(self.min_alignment),
        }
    }

    /// Extract co-access information from patterns.
    fn extract_co_accesses(
        &self,
        fields: &[Field],
        patterns: &HashMap<String, AccessPattern>,
    ) -> Vec<(String, String, f64)> {
        let mut result = Vec::new();
        let field_names: HashSet<_> = fields.iter().map(|f| f.name.as_str()).collect();

        for pattern in patterns.values() {
            for co in &pattern.co_accesses {
                // Extract field name from qualified name (Type.field)
                let a = co.field_a.split('.').next_back().unwrap_or(&co.field_a);
                let b = co.field_b.split('.').next_back().unwrap_or(&co.field_b);

                if field_names.contains(a) && field_names.contains(b) {
                    result.push((a.to_string(), b.to_string(), co.correlation));
                }
            }
        }

        result
    }

    /// Build groups from co-access information.
    fn build_groups(
        &self,
        fields: &[Field],
        co_accesses: &[(String, String, f64)],
    ) -> Vec<FieldGroup> {
        self.build_groups_simple(fields, co_accesses)
    }

    /// Build groups with simple co-access list.
    fn build_groups_simple(
        &self,
        fields: &[Field],
        co_accesses: &[(String, String, f64)],
    ) -> Vec<FieldGroup> {
        // Union-find for grouping
        let mut parent: HashMap<&str, &str> = HashMap::new();

        for field in fields {
            parent.insert(&field.name, &field.name);
        }

        fn find<'a>(parent: &mut HashMap<&'a str, &'a str>, x: &'a str) -> &'a str {
            if parent.get(x).copied() != Some(x) {
                let p = parent.get(x).copied().unwrap_or(x);
                let root = find(parent, p);
                parent.insert(x, root);
                root
            } else {
                x
            }
        }

        // Merge strongly co-accessed fields
        for (a, b, corr) in co_accesses {
            if *corr >= 0.5 {
                let a_str = a.as_str();
                let b_str = b.as_str();

                if parent.contains_key(a_str) && parent.contains_key(b_str) {
                    let ra = find(&mut parent, a_str);
                    let rb = find(&mut parent, b_str);
                    if ra != rb {
                        parent.insert(ra, rb);
                    }
                }
            }
        }

        // Build groups from union-find
        let mut group_map: HashMap<&str, FieldGroup> = HashMap::new();
        let field_sizes: HashMap<&str, usize> =
            fields.iter().map(|f| (f.name.as_str(), f.size)).collect();

        for field in fields {
            let root = find(&mut parent, &field.name);
            let group = group_map.entry(root).or_default();
            group.add(&field.name, field.size);
        }

        // Add correlation info
        for (a, b, corr) in co_accesses {
            if let Some(root) = parent.get(a.as_str())
                && let Some(group) = group_map.get_mut(*root)
            {
                group.correlation = group.correlation.max(*corr);
            }
        }

        group_map.into_values().collect()
    }

    /// Compute field ordering.
    fn compute_ordering(
        &self,
        fields: &[Field],
        groups: &[FieldGroup],
        patterns: &HashMap<String, AccessPattern>,
    ) -> Vec<usize> {
        // Compute hotness for each field
        let mut hotness: HashMap<&str, f64> = HashMap::new();

        for pattern in patterns.values() {
            for access in &pattern.accesses {
                let field = access
                    .field_name
                    .split('.')
                    .next_back()
                    .unwrap_or(&access.field_name);
                let heat = access.heat();
                let current = *hotness.get(field).unwrap_or(&0.0);
                hotness.insert(field, current.max(heat));
            }
        }

        // Sort by group first, then by hotness within group
        let mut indices: Vec<usize> = (0..fields.len()).collect();

        if self.hot_first {
            indices.sort_by(|&a, &b| {
                let ha = hotness.get(fields[a].name.as_str()).unwrap_or(&0.0);
                let hb = hotness.get(fields[b].name.as_str()).unwrap_or(&0.0);
                hb.partial_cmp(ha).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Within groups, keep fields together
        self.group_order_fields(fields, &indices, groups)
    }

    /// Compute simple ordering without patterns.
    fn compute_simple_ordering(&self, fields: &[Field], groups: &[FieldGroup]) -> Vec<usize> {
        let indices: Vec<usize> = (0..fields.len()).collect();
        self.group_order_fields(fields, &indices, groups)
    }

    /// Order fields keeping groups together.
    fn group_order_fields(
        &self,
        fields: &[Field],
        initial_order: &[usize],
        groups: &[FieldGroup],
    ) -> Vec<usize> {
        let mut result = Vec::new();
        let mut used: HashSet<usize> = HashSet::new();

        // Map field names to indices
        let name_to_idx: HashMap<&str, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.as_str(), i))
            .collect();

        // Process in initial order, but keep group members together
        for &idx in initial_order {
            if used.contains(&idx) {
                continue;
            }

            let field = &fields[idx];

            // Find which group this field belongs to
            for group in groups {
                if group.fields.contains(&field.name) {
                    // Add all group members
                    for member in &group.fields {
                        if let Some(&member_idx) = name_to_idx.get(member.as_str())
                            && !used.contains(&member_idx)
                        {
                            result.push(member_idx);
                            used.insert(member_idx);
                        }
                    }
                    break;
                }
            }

            // If not in any group, add individually
            if !used.contains(&idx) {
                result.push(idx);
                used.insert(idx);
            }
        }

        result
    }

    /// Compute the final layout.
    fn compute_layout(
        &self,
        fields: &[Field],
        ordering: &[usize],
        groups: &[FieldGroup],
    ) -> PackedLayout {
        let struct_name = "Packed"; // Would be passed in real usage
        let mut layout = PackedLayout::new(struct_name);
        layout.groups = groups.to_vec();

        let mut offset = 0usize;
        let mut total_padding = 0usize;

        for &idx in ordering {
            let field = &fields[idx];

            // Align
            let align = field.alignment;
            let aligned_offset = offset.div_ceil(align) * align;
            total_padding += aligned_offset - offset;
            offset = aligned_offset;

            // Compute cache line info
            let cache_line = offset / self.cache_line_size;
            let end_cache_line = (offset + field.size - 1) / self.cache_line_size;
            let spans = cache_line != end_cache_line;

            layout.fields.push(PackedField {
                name: field.name.clone(),
                offset,
                size: field.size,
                cache_line,
                spans_cache_line: spans,
                original_position: field.original_position,
            });

            offset += field.size;
        }

        // Final alignment for struct
        let struct_align = fields.iter().map(|f| f.alignment).max().unwrap_or(1);
        let final_size = offset.div_ceil(struct_align) * struct_align;
        total_padding += final_size - offset;

        layout.total_size = final_size;
        layout.padding = total_padding;
        layout.cache_lines = final_size.div_ceil(self.cache_line_size);

        // Compute improvement (based on cache line reduction)
        let original_cache_lines = self.estimate_original_cache_lines(fields);
        if original_cache_lines > 0 {
            layout.improvement = 1.0 - (layout.cache_lines as f64 / original_cache_lines as f64);
        }

        layout
    }

    /// Estimate cache lines for original layout.
    fn estimate_original_cache_lines(&self, fields: &[Field]) -> usize {
        let mut offset = 0usize;

        for field in fields {
            let aligned = offset.div_ceil(field.alignment) * field.alignment;
            offset = aligned + field.size;
        }

        offset.div_ceil(self.cache_line_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = Field::new("x", 8, 8)
            .at_position(0)
            .with_semantic_type("f64");

        assert_eq!(field.name, "x");
        assert_eq!(field.size, 8);
        assert_eq!(field.original_position, 0);
        assert_eq!(field.semantic_type, Some("f64".to_string()));
    }

    #[test]
    fn test_field_group() {
        let mut group = FieldGroup::new();
        group.add("x", 8);
        group.add("y", 8);
        group.add("z", 8);

        assert_eq!(group.total_size, 24);
        assert!(group.fits_in_cache_line(64));
        assert!(!group.fits_in_cache_line(16));
    }

    #[test]
    fn test_simple_packing() {
        let packer = CacheLinePacker::new(64);

        let fields = [
            ("a", 1), // 1 byte
            ("b", 8), // 8 bytes (will create padding after a)
            ("c", 1), // 1 byte
            ("d", 4), // 4 bytes
        ];

        let co_accesses = [
            (("a", "c"), 0.9), // a and c are accessed together
        ];

        let layout = packer.pack_simple(&fields, &co_accesses);

        // a and c should be grouped together
        assert!(layout.total_size > 0);
        assert!(layout.cache_lines >= 1);
    }

    #[test]
    fn test_layout_comments() {
        let mut layout = PackedLayout::new("TestStruct");
        layout.total_size = 24;
        layout.cache_lines = 1;
        layout.padding = 2;
        layout.improvement = 0.5;

        layout.fields.push(PackedField {
            name: "x".to_string(),
            offset: 0,
            size: 8,
            cache_line: 0,
            spans_cache_line: false,
            original_position: 0,
        });

        let comments = layout.to_comments();
        assert!(comments.contains("TestStruct"));
        assert!(comments.contains("24 bytes"));
    }

    #[test]
    fn test_same_cache_line() {
        let mut layout = PackedLayout::new("Test");

        layout.fields.push(PackedField {
            name: "a".to_string(),
            offset: 0,
            size: 8,
            cache_line: 0,
            spans_cache_line: false,
            original_position: 0,
        });

        layout.fields.push(PackedField {
            name: "b".to_string(),
            offset: 8,
            size: 8,
            cache_line: 0,
            spans_cache_line: false,
            original_position: 1,
        });

        layout.fields.push(PackedField {
            name: "c".to_string(),
            offset: 64,
            size: 8,
            cache_line: 1,
            spans_cache_line: false,
            original_position: 2,
        });

        assert!(layout.same_cache_line("a", "b"));
        assert!(!layout.same_cache_line("a", "c"));
    }

    #[test]
    fn test_packer_without_hot_first() {
        let packer = CacheLinePacker::new(64).without_hot_first();

        let fields = [("a", 8), ("b", 8)];
        let layout = packer.pack_simple(&fields, &[]);

        assert_eq!(layout.fields.len(), 2);
    }

    #[test]
    fn test_span_detection() {
        let packer = CacheLinePacker::new(64);

        // Create a field that will span cache lines
        let fields = [
            ("big", 128), // 128 bytes will definitely span
        ];

        let layout = packer.pack_simple(&fields, &[]);

        assert!(layout.fields[0].spans_cache_line);
    }

    #[test]
    fn test_compute_alignment() {
        let packer = CacheLinePacker::new(64);

        assert_eq!(packer.compute_alignment(1), 1);
        assert_eq!(packer.compute_alignment(2), 2);
        assert_eq!(packer.compute_alignment(4), 4);
        assert_eq!(packer.compute_alignment(8), 8);
        assert_eq!(packer.compute_alignment(16), 8);
    }

    #[test]
    fn test_offset_of() {
        let mut layout = PackedLayout::new("Test");

        layout.fields.push(PackedField {
            name: "x".to_string(),
            offset: 0,
            size: 8,
            cache_line: 0,
            spans_cache_line: false,
            original_position: 0,
        });

        layout.fields.push(PackedField {
            name: "y".to_string(),
            offset: 8,
            size: 4,
            cache_line: 0,
            spans_cache_line: false,
            original_position: 1,
        });

        assert_eq!(layout.offset_of("x"), Some(0));
        assert_eq!(layout.offset_of("y"), Some(8));
        assert_eq!(layout.offset_of("z"), None);
    }
}
