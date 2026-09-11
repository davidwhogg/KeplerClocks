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

DROP TABLE IF EXISTS clock;
CREATE TABLE clock (
    -- mode_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
    clock_id INTEGER PRIMARY KEY AUTOINCREMENT,
    omega DOUBLE NOT NULL,
    degree INTEGER NOT NULL,
    star_id VARCHAR(32) NOT NULL,
    dataset_id VARCHAR(32) NOT NULL,
    empirical_value DOUBLE,
    theoretical_value DOUBLE,
    timing_precision_jackknife DOUBLE,
    timing_precision_split DOUBLE
);
