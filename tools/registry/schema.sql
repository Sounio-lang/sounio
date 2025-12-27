-- Sounio Package Registry Database Schema
-- PostgreSQL 14+
--
-- This schema defines the database structure for the Sounio package registry.
-- It includes tables for users, packages, versions, downloads, and authentication.

-- =============================================================================
-- Extensions
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- =============================================================================
-- Users
-- =============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(39) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    bio TEXT,
    location VARCHAR(255),
    website VARCHAR(255),
    github_username VARCHAR(39),
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    is_suspended BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT username_format CHECK (username ~ '^[a-zA-Z0-9_-]+$'),
    CONSTRAINT email_format CHECK (email ~ '^[^@]+@[^@]+\.[^@]+$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);

-- User email verification tokens
CREATE TABLE email_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_email_verifications_user_id ON email_verifications(user_id);
CREATE INDEX idx_email_verifications_token ON email_verifications(token);

-- Password reset tokens
CREATE TABLE password_resets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_password_resets_token ON password_resets(token);

-- =============================================================================
-- API Tokens
-- =============================================================================

CREATE TABLE api_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    token_hash VARCHAR(255) NOT NULL UNIQUE,  -- SHA256 hash of token
    scopes TEXT[] NOT NULL DEFAULT ARRAY['read', 'publish'],
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    last_used_ip INET,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_scopes CHECK (
        scopes <@ ARRAY['read', 'publish', 'admin']::TEXT[]
    )
);

CREATE INDEX idx_api_tokens_user_id ON api_tokens(user_id);
CREATE INDEX idx_api_tokens_token_hash ON api_tokens(token_hash);

-- =============================================================================
-- Categories
-- =============================================================================

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    package_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default categories
INSERT INTO categories (slug, name, description) VALUES
    ('algorithms', 'Algorithms', 'Data structures and algorithms'),
    ('async', 'Async', 'Async runtime and utilities'),
    ('cli', 'CLI', 'Command-line tools and utilities'),
    ('concurrency', 'Concurrency', 'Concurrent programming'),
    ('cryptography', 'Cryptography', 'Cryptographic algorithms and utilities'),
    ('data-science', 'Data Science', 'Data analysis and machine learning'),
    ('database', 'Database', 'Database drivers and ORMs'),
    ('development', 'Development', 'Developer tools and utilities'),
    ('ffi', 'FFI', 'Foreign function interface'),
    ('graphics', 'Graphics', 'Graphics and visualization'),
    ('gui', 'GUI', 'Graphical user interfaces'),
    ('math', 'Math', 'Mathematics and numerical computing'),
    ('network', 'Network', 'Networking and protocols'),
    ('no-std', 'No-std', 'Libraries that work without standard library'),
    ('os', 'Operating System', 'OS-specific functionality'),
    ('parser', 'Parser', 'Parsing and serialization'),
    ('science', 'Science', 'Scientific computing'),
    ('text', 'Text', 'Text processing'),
    ('web', 'Web', 'Web development'),
    ('wasm', 'WASM', 'WebAssembly');

-- =============================================================================
-- Packages
-- =============================================================================

CREATE TABLE packages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(64) NOT NULL UNIQUE,
    description TEXT,
    homepage VARCHAR(255),
    repository VARCHAR(255),
    documentation VARCHAR(255),
    license VARCHAR(64),
    readme TEXT,
    keywords TEXT[],
    downloads BIGINT NOT NULL DEFAULT 0,
    recent_downloads BIGINT NOT NULL DEFAULT 0,  -- Last 90 days
    is_yanked BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT name_format CHECK (name ~ '^[a-z][a-z0-9_-]*$'),
    CONSTRAINT name_length CHECK (LENGTH(name) BETWEEN 1 AND 64)
);

CREATE INDEX idx_packages_name ON packages(name);
CREATE INDEX idx_packages_downloads ON packages(downloads DESC);
CREATE INDEX idx_packages_created_at ON packages(created_at);
CREATE INDEX idx_packages_updated_at ON packages(updated_at);
CREATE INDEX idx_packages_keywords ON packages USING gin(keywords);
CREATE INDEX idx_packages_name_trgm ON packages USING gin(name gin_trgm_ops);
CREATE INDEX idx_packages_description_trgm ON packages USING gin(description gin_trgm_ops);

-- Package categories (many-to-many)
CREATE TABLE package_categories (
    package_id UUID NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (package_id, category_id)
);

CREATE INDEX idx_package_categories_category ON package_categories(category_id);

-- Package owners
CREATE TABLE package_owners (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    package_id UUID NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    added_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(package_id, user_id)
);

CREATE INDEX idx_package_owners_package ON package_owners(package_id);
CREATE INDEX idx_package_owners_user ON package_owners(user_id);

-- =============================================================================
-- Versions
-- =============================================================================

CREATE TABLE versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    package_id UUID NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    version VARCHAR(64) NOT NULL,
    checksum VARCHAR(64) NOT NULL,  -- SHA256
    size BIGINT NOT NULL,  -- Bytes
    downloads BIGINT NOT NULL DEFAULT 0,
    features JSONB,
    rust_version VARCHAR(32),  -- Minimum Sounio compiler version
    readme TEXT,
    changelog TEXT,
    is_yanked BOOLEAN NOT NULL DEFAULT FALSE,
    yanked_at TIMESTAMPTZ,
    yanked_by UUID REFERENCES users(id),
    yanked_reason TEXT,
    published_by UUID NOT NULL REFERENCES users(id),
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(package_id, version)
);

CREATE INDEX idx_versions_package ON versions(package_id);
CREATE INDEX idx_versions_published_at ON versions(published_at);
CREATE INDEX idx_versions_downloads ON versions(downloads DESC);

-- Version dependencies
CREATE TABLE dependencies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id UUID NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    package_name VARCHAR(64) NOT NULL,
    version_req VARCHAR(255) NOT NULL,
    features TEXT[],
    optional BOOLEAN NOT NULL DEFAULT FALSE,
    kind VARCHAR(16) NOT NULL DEFAULT 'normal',  -- normal, dev, build

    CONSTRAINT valid_kind CHECK (kind IN ('normal', 'dev', 'build'))
);

CREATE INDEX idx_dependencies_version ON dependencies(version_id);
CREATE INDEX idx_dependencies_package_name ON dependencies(package_name);

-- Reverse dependencies (cached for performance)
CREATE TABLE reverse_dependencies (
    package_id UUID NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    dependent_package_id UUID NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    version_count INTEGER NOT NULL DEFAULT 1,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (package_id, dependent_package_id)
);

CREATE INDEX idx_reverse_deps_dependent ON reverse_dependencies(dependent_package_id);

-- =============================================================================
-- Downloads Tracking
-- =============================================================================

CREATE TABLE downloads (
    id BIGSERIAL PRIMARY KEY,
    version_id UUID NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    downloaded_at DATE NOT NULL DEFAULT CURRENT_DATE,
    count BIGINT NOT NULL DEFAULT 1,

    UNIQUE(version_id, downloaded_at)
);

CREATE INDEX idx_downloads_version ON downloads(version_id);
CREATE INDEX idx_downloads_date ON downloads(downloaded_at);

-- Daily aggregated downloads per package
CREATE TABLE package_downloads_daily (
    package_id UUID NOT NULL REFERENCES packages(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    count BIGINT NOT NULL DEFAULT 0,

    PRIMARY KEY (package_id, date)
);

CREATE INDEX idx_package_downloads_daily_date ON package_downloads_daily(date);

-- =============================================================================
-- Audit Log
-- =============================================================================

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(64) NOT NULL,
    resource_type VARCHAR(32) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);

-- =============================================================================
-- Functions
-- =============================================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_packages_updated_at
    BEFORE UPDATE ON packages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Increment package download count
CREATE OR REPLACE FUNCTION increment_downloads(p_version_id UUID)
RETURNS VOID AS $$
DECLARE
    v_package_id UUID;
BEGIN
    -- Get package ID
    SELECT package_id INTO v_package_id FROM versions WHERE id = p_version_id;

    -- Update version downloads
    UPDATE versions SET downloads = downloads + 1 WHERE id = p_version_id;

    -- Update package downloads
    UPDATE packages SET downloads = downloads + 1, recent_downloads = recent_downloads + 1
    WHERE id = v_package_id;

    -- Update or insert daily download count
    INSERT INTO downloads (version_id, downloaded_at, count)
    VALUES (p_version_id, CURRENT_DATE, 1)
    ON CONFLICT (version_id, downloaded_at)
    DO UPDATE SET count = downloads.count + 1;

    -- Update package daily downloads
    INSERT INTO package_downloads_daily (package_id, date, count)
    VALUES (v_package_id, CURRENT_DATE, 1)
    ON CONFLICT (package_id, date)
    DO UPDATE SET count = package_downloads_daily.count + 1;
END;
$$ LANGUAGE plpgsql;

-- Update category package count
CREATE OR REPLACE FUNCTION update_category_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE categories SET package_count = package_count + 1 WHERE id = NEW.category_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE categories SET package_count = package_count - 1 WHERE id = OLD.category_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_category_count
    AFTER INSERT OR DELETE ON package_categories
    FOR EACH ROW
    EXECUTE FUNCTION update_category_count();

-- Search packages
CREATE OR REPLACE FUNCTION search_packages(
    p_query TEXT,
    p_category_id INTEGER DEFAULT NULL,
    p_limit INTEGER DEFAULT 20,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    id UUID,
    name VARCHAR(64),
    description TEXT,
    downloads BIGINT,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.name,
        p.description,
        p.downloads,
        (
            similarity(p.name, p_query) * 3 +
            similarity(COALESCE(p.description, ''), p_query)
        ) AS rank
    FROM packages p
    LEFT JOIN package_categories pc ON p.id = pc.package_id
    WHERE
        (p.name ILIKE '%' || p_query || '%' OR
         p.description ILIKE '%' || p_query || '%' OR
         p_query = ANY(p.keywords))
        AND (p_category_id IS NULL OR pc.category_id = p_category_id)
        AND NOT p.is_yanked
    GROUP BY p.id
    ORDER BY rank DESC, p.downloads DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Views
-- =============================================================================

-- Latest version for each package
CREATE VIEW latest_versions AS
SELECT DISTINCT ON (package_id)
    id,
    package_id,
    version,
    checksum,
    size,
    downloads,
    features,
    rust_version,
    is_yanked,
    published_by,
    published_at
FROM versions
WHERE NOT is_yanked
ORDER BY package_id, published_at DESC;

-- Package summary (for listings)
CREATE VIEW package_summaries AS
SELECT
    p.id,
    p.name,
    p.description,
    p.homepage,
    p.repository,
    p.license,
    p.keywords,
    p.downloads,
    p.recent_downloads,
    p.created_at,
    p.updated_at,
    lv.version AS latest_version,
    lv.published_at AS latest_published_at,
    array_agg(DISTINCT c.slug) FILTER (WHERE c.slug IS NOT NULL) AS categories,
    array_agg(DISTINCT u.username) FILTER (WHERE u.username IS NOT NULL) AS owners
FROM packages p
LEFT JOIN latest_versions lv ON p.id = lv.package_id
LEFT JOIN package_categories pc ON p.id = pc.package_id
LEFT JOIN categories c ON pc.category_id = c.id
LEFT JOIN package_owners po ON p.id = po.package_id
LEFT JOIN users u ON po.user_id = u.id
WHERE NOT p.is_yanked
GROUP BY p.id, lv.version, lv.published_at;

-- =============================================================================
-- Scheduled Jobs (use pg_cron or external scheduler)
-- =============================================================================

-- Refresh recent downloads (run daily)
-- UPDATE packages SET recent_downloads = (
--     SELECT COALESCE(SUM(d.count), 0)
--     FROM downloads d
--     JOIN versions v ON d.version_id = v.id
--     WHERE v.package_id = packages.id
--     AND d.downloaded_at > CURRENT_DATE - INTERVAL '90 days'
-- );

-- Clean up expired tokens (run daily)
-- DELETE FROM api_tokens WHERE expires_at < NOW();

-- Clean up old audit logs (run monthly)
-- DELETE FROM audit_log WHERE created_at < NOW() - INTERVAL '1 year';
