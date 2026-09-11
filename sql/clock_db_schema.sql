-- DROP USER 'nana'@'localhost';
-- FLUSH PRIVILEGES;
-- CREATE USER 'nana'@'%';

-- CREATE DATABASE IF NOT EXISTS stars_db;
-- USE stars_db;
-- GRANT ALL ON stars_db.* TO 'nana'@'%';


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

DROP TABLE IF EXISTS mode;
CREATE TABLE mode (
    -- mode_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
    mode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frequency DOUBLE NOT NULL,
    star_id VARCHAR(32) NOT NULL,
    dataset_id VARCHAR(32) NOT NULL,
    region VARCHAR(1),
    amplitude_a DOUBLE,
    amplitude_b DOUBLE,
    delta_chi_squared DOUBLE,
    frequency_region_A DOUBLE,
    phase_uncertainty_jackknife DOUBLE,
    phase_uncertainty_split DOUBLE,
    parent_mode_id INTEGER NULL  
);

DROP TABLE IF EXISTS amplitude;
CREATE TABLE amplitude (
    amplitude_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode_id INTEGER NOT NULL,
    harmonic INTEGER NOT NULL,
    amplitude_a DOUBLE,
    amplitude_b DOUBLE,
    FOREIGN KEY (mode_id) REFERENCES mode(mode_id)
);

DROP VIEW IF EXISTS parent_mode;
CREATE VIEW parent_mode AS
SELECT 
    m.mode_id,
    m.star_id,
    m.frequency as parent_frequency,
    COUNT(children.mode_id) as num_of_children,
    0.5 * SUM(a.amplitude_a * a.amplitude_a + a.amplitude_b * a.amplitude_b) as variance
FROM mode m
LEFT JOIN mode children ON children.parent_mode_id = m.mode_id
LEFT JOIN amplitude a ON a.mode_id = children.mode_id AND a.harmonic = 1
WHERE m.parent_mode_id IS NULL
GROUP BY m.mode_id;
