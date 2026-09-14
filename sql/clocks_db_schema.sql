-- # Schema for the KeplerClocks project

DROP TABLE IF EXISTS dataset;
CREATE TABLE IF NOT EXISTS dataset (
    dataset_id VARCHAR(32),
    description TEXT,
    PRIMARY KEY (dataset_id)
);

DROP TABLE IF EXISTS star;
CREATE TABLE IF NOT EXISTS star (
    star_id VARCHAR(32),
    gaia_ra DOUBLE,
    gaia_dec DOUBLE,
    gaia_g DOUBLE,
    gaia_br DOUBLE,
    gaia_parallax DOUBLE,
    total_observing_time DOUBLE,
    PRIMARY KEY (star_id)
);

DROP TABLE IF EXISTS task;
CREATE TABLE IF NOT EXISTS task (
    star_id VARCHAR(32),
    dataset_id VARCHAR(32),
    process_id VARCHAR(32) NULL,
    code_version VARCHAR(32) NULL,
    started TIMESTAMP NULL,
    finished TIMESTAMP NULL,
    message VARCHAR(256) NULL,
    error_message TEXT NULL,
    PRIMARY KEY (star_id, dataset_id)
);

DROP TABLE IF EXISTS clock;
CREATE TABLE clock (
    -- mode_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
    clock_id INTEGER PRIMARY KEY AUTOINCREMENT,
    angular_frequency DOUBLE NOT NULL,
    fourier_series_degree INTEGER NOT NULL,
    star_id VARCHAR(32) NOT NULL,
    dataset_id VARCHAR(32) NOT NULL,
    empirical_value DOUBLE,
    theoretical_value DOUBLE,
    timing_precision_jackknife DOUBLE,
    timing_precision_split DOUBLE
);

DROP VIEW IF EXISTS best_clock;
CREATE VIEW best_clock AS
WITH ranked_clocks AS (
    SELECT 
        *,
        COUNT(*) OVER(PARTITION BY star_id) AS best_of,
        ROW_NUMBER() OVER(PARTITION BY star_id ORDER BY theoretical_value DESC) AS rn
    FROM clock
)
SELECT 
    star_id, best_of, angular_frequency, fourier_series_degree, empirical_value, theoretical_value
FROM ranked_clocks
WHERE rn = 1;
