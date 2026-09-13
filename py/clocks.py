"""
# clocks.py
Functions for finding coherent clocks in NASA *Kepler* light curves.

## authors:
- **Nana Miller** (NYU)
- **David W. Hogg** (NYU)

## license:
- Copyright 2026 the authors. This code is licensed for re-use under the *MIT License*.

## bugs and issues and to-do items:
- Needs a function "do KICID" that just clears the data on a single KICID and runs it.
- Some nomenclature is bad about star vs light curve. KICID is a star, KICID + long is a light curve?
- This code needs some Jupyter notebooks that can be used to test sub-parts. Development is bad rn.
- Ought to subtract some fiducial BJD for numerical stability.

## notes:
- This code doesn't even import matplotlib; plotting must be done elsewhere.

## calling sequence:
- `python clocks.py db` #creates the database and fills the task table
- `nohup python clocks.py worker > q.log 2>&1 &` #runs a worker in the background
"""
from functools import partial
from fractions import Fraction
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.signal import find_peaks

import lightkurve as lk
from astropy.timeseries import LombScargle
from astropy import units as u
from astropy.table import Table
from astropy.io import ascii
from astropy.time import Time, TimeDelta

import time
import sqlite3 as db
import os
import sys

# set constants
MAX_NUMBER_OF_STARS = 500 # for testing! raise this to 300_000 for production
MIN_NUMBER_OF_MEASUREMENTS = 10_000
CLOCKS_DB_FILE = "../data/clocks.db"
MAX_PERIOD = 30. # days
MIN_THEORETICAL_VALUE = 1.e9 # inverse days squared
MAX_INTRASTAR_VALUE_RATIO = 1.e3

def get_kepler_data(kic_id, exptime='long'):
    
    start = time.time()
    """
    ## Inputs:
    `kic_id`: Kepler ID (str)
    `exptime`: default exposure time, 'long'

    ## Outputs:
    Returns a 5-tuple (or `None` if failed):
    - `times`, `fluxes`, `errors`:  light curve (days, dimensionless, dimensionless)
    - `delta_f`: frequency resolution, 1 / [total observation time] (inverse days)
    - `sampling_time`: time resolution [median time between observations] (days)

    ## Bugs:
    - Way too much `try` / `except`.
    - Dependent on slightly unreliable lightkurve data caching; ought to clear cache on bad objects.
    """
    print("clocks.get_kepler_data(): starting to download data for", kic_id)
    try:
        search_result = lk.search_lightcurve(kic_id, mission = 'Kepler', exptime=exptime)
        if len(search_result) < 1:
            msg = f"clocks.get_kepler_data: no results for {kic_id} at this cadence"
            print(msg)
            update_error_message(kic_id, 'Kepler_long', msg)
            return None
    except Exception as e:
        print(f"Exception for {kic_id}: get_kepler_data(): lk.search_lightcurve() failed for {kic_id } with {str(e)}")
        update_error_message(kic_id, 'Kepler_long', str(e))
        return None
    try:
        lc_collection = search_result.download_all()
    except lk.LightkurveError as e:
        print(f"LightkurveError for {kic_id}: get_kepler_data(): search_result.download_all() failed for {kic_id} with {str(e)}")
        update_error_message(kic_id, 'Kepler_long', str(e))
        return None
    except Exception as e:
        print(f"Exception for {kic_id}: get_kepler_data(): search_result.download_all() failed for {kic_id} with {str(e)}")
        update_error_message(kic_id, 'Kepler_long', str(e))
        return None
    try:
        lc = lc_collection.stitch()
        #print("get_kepler_data(): minimum time value", np.min(lc.time.value), np.min(lc.time), lc.time)
    except lk.LightkurveError as e:
        print(f"LightkurveError for {kic_id}: get_kepler_data(): lc_collection.stitch() failed for {kic_id} with {str(e)}")
        update_error_message(kic_id, 'Kepler_long', str(e))
        return None
    except Exception as e:
        print(f"Exception for {kic_id}: get_kepler_data(): lc_collection.stitch() failed for {kic_id} with {str(e)}")
        update_error_message(kic_id, 'Kepler_long', str(e))
        return None
        
    # unpack, remove bad data
    times, fluxes, errors = lc.time.value, lc.flux.value, lc.flux_err.value
    good = np.isfinite(times) & np.isfinite(fluxes) & np.isfinite(errors)
    times, fluxes, errors = times[good], fluxes[good], errors[good]
    if len(times) < MIN_NUMBER_OF_MEASUREMENTS:
        msg = f"clocks.get_kepler_data: not enough data from Kepler on {kic_id} at this cadence"
        print(msg)
        update_error_message(kic_id, 'Kepler_long', msg)
        return None

    # reorder
    idx = np.argsort(times)
    times, fluxes, errors = times[idx], fluxes[idx], errors[idx]
    delta_f = (1/(times[-1] - times[0]))
    sampling_time= np.median(np.diff(times))

    print("clocks.get_kepler_data() took", time.time() - start, "s")
    return times, fluxes, errors, delta_f, sampling_time

def get_candidate_frequencies(ts, ys, errs, df, dt, max_peaks=32, nterms=8):
    """
    # inputs:
    - `ts`, `ys`, `errs`: the light curve
    - `df`, `dt`: the smallest frequency and the smallest time of relevance

    # bugs:
    - The `nterms` input is set without significant testing.
    - It is not clear that 32 is the right number of max peaks for default.
    - Should this code multiply by `fs ** 2` before peak-finding? To make it more like a value?
    - MAGIC oversampling by a factor of either 2 or 4 (I don't know which)
    """
    fs = np.arange(1. / MAX_PERIOD, 0.5 / dt, 0.25 * df)
    ps = LombScargle(ts, ys, errs, nterms=nterms).power(fs)
    idxs, _ = find_peaks(ps, distance=4)
    ii = np.argsort(ps[idxs])[::-1]
    idxs = idxs[ii]
    if len(idxs) > max_peaks:
        idxs = idxs[:max_peaks]
    return fs[idxs]

@partial(jax.jit, static_argnums=1)
def design_matrix(om, M, t,):
    """
    bug: Doesn't use jax-finufft?
    """
    ms1, ms2 = jnp.arange(M + 1), jnp.arange(1, M + 1)
    return jnp.concat((jnp.cos(ms1[None, :] * om * t[:, None]),
                       jnp.sin(ms2[None, :] * om * t[:, None])), axis=1), \
           jnp.concat((ms1, ms2))

@partial(jax.jit, static_argnums=1)
def fourier_wls_fit(om, M, t, y, iv):
    X, m = design_matrix(om, M, t)
    return X, m, jnp.linalg.solve(X.T @ (iv[:, None] * X),
                                  X.T @ (iv * y))

@partial(jax.jit, static_argnums=1)
def clock_value(om, M, t, y, iv):
    """
    # inputs:
    - `om`: frequency to test
    - `t`, `y`, `iv`: light curve (iv is inverse variance, not error)
    - `M`: degree of Fourier series

    # bugs:
    - This is very affected by outliers; need to remove those somehow. But *don't* use `jnp.median()`!
    - Maybe we should use IRLS to do the fit.
    - Maybe we should penalize (or increase) MSE according to model complexity `2 * M + 1`.
    """
    X, m, pars = fourier_wls_fit(om, M, t, y, iv)
    mse = jnp.sum(iv * (y - X @ pars) ** 2) / jnp.sum(iv) # weighted mean
    return len(y) * (om ** 2 / mse) * jnp.sum(m ** 2 * pars ** 2)

clock_values = jax.vmap(clock_value, in_axes=(0, None, None, None, None))

def optimistic_clock_value(om, M, t, y, iv):
    _, m, pars = fourier_wls_fit(om, M, t, y, iv)
    return np.sum(iv) * om ** 2 * jnp.sum(m ** 2 * pars ** 2)

def take_derivative_wrt_phase(ps, ms):
    M = (len(ms) - 1) // 2
    newps = np.zeros_like(ps)
    newps[1 : M + 1] = ms[M + 1 :] * ps[M + 1 :]
    newps[M + 1 :] = -1. * ms[1 : M + 1] * ps[1 : M + 1]
    return newps

def theoretical_clock_value(om, M, t, y, iv):
    X, m, pars = fourier_wls_fit(om, M, t, y, iv)
    dpars = take_derivative_wrt_phase(pars, m)
    derivs = X @ dpars
    return om ** 2 * np.sum(iv * derivs ** 2)

def get_best_clock(om0, Mmax, t, y, iv, df, dt):
    """
    # inputs:
    - `om0`: first guess at a good clock (angular) frequency omega
    - `t`, `y`, `iv`: the light curve
    - `Mmax`: the maximum degree of the Fourier series (not necessarily the degree)
    - `df`: the frequency resolution (non-angular frequency) in the data
    - `dt`: the sampling time (related to Nyquist).

    # bugs:
    - Takes one input as angular frequency and another as frequency.
    - MAGIC 0.05

    # notes:
    - Recursion is insane.
    - Works in the log for stability.
    """
    nyquist = np.pi / dt # angular-frequency units
    M = max(1, min(Mmax, int(nyquist // om0)))
    do = 0.05 * np.pi * df # magic 0.05
    oms = np.array([om0 - do, om0, om0 + do])
    ys = np.log(clock_values(oms, M, t, y, iv))
    if np.argmax(ys) != 1:
        return get_best_clock(oms[np.argmax(ys)], M, t, y, iv, df, dt)
    ii = jnp.argmax(ys)
    foo = jnp.polyfit(oms, ys, 2)
    om = jnp.roots(jnp.polyder(foo), strip_zeros=False).real
    return om[0], M, jnp.exp(jnp.polyval(foo, om))[0]

def identify_resonances(fs, fres, max_denominator=12):
    """
    ## inputs:
    - `fs`: list of frequencies, ordered from most important to least.
    - `fres`: expected frequency resolution (like 1 / [total time] or something like that)

    ## notes:
    - This can be run on angular frequencies or frequencies. Just make sure that `fs` and `fres` have the same units!
    
    ## bugs:
    - REQUIRES that the `fs` be ordered from highest value to lowest.
    """
    duplicates = np.zeros_like(fs).astype(bool)
    for i, f in enumerate(fs):
        if not duplicates[i]:
            for j in range(i + 1, len(fs)):
                tol = fres / min(fs[i], fs[j])
                ratio_ji = fs[j] / fs[i]
                frac_ji = Fraction(ratio_ji).limit_denominator(max_denominator)
                test_ji = abs((ratio_ji - float(frac_ji)) / ratio_ji) < tol
                ratio_ij = fs[i] / fs[j]
                frac_ij = Fraction(ratio_ij).limit_denominator(max_denominator)
                test_ij = abs((ratio_ij - float(frac_ij)) / ratio_ij) < tol
                duplicates[j] = test_ji | test_ij
                print("identify_resonances():", fs[i], fs[j], ratio_ji, frac_ji, ratio_ij, frac_ij, duplicates[j])
    return duplicates

def best_clocks_in_star(kicid, Mmax=128, plot=True):
    print(f"clocks.best_clocks_in_star(): getting Kepler data for {kicid}")
    foo = get_kepler_data(kicid)
    if foo is None:
        print(f"clocks.best_clocks_in_star(): skipping {kicid}")
        return None
    ts, ys, errs, deltaf, deltat = foo
    ivars = 1. / errs ** 2
    print(f"clocks.best_clocks_in_star(): getting candidate frequencies for {kicid}")
    candidate_oms = 2. * np.pi * get_candidate_frequencies(ts, ys, errs, deltaf, deltat)

    # now loop through candidates and refine them
    oms, values = np.zeros_like(candidate_oms), np.zeros_like(candidate_oms)
    Ms = np.zeros_like(candidate_oms).astype(int)
    print(f"clocks.best_clocks_in_star(): getting clock values for {kicid}")
    for i, om0 in enumerate(candidate_oms):
        # BUG: Should be a map
        oms[i], Ms[i], values[i] = get_best_clock(om0, Mmax, ts, ys, ivars, deltaf, deltat)

    # now build and populate an astropy table
    clocks = Table([oms, Ms, values], names=('angular_frequency', 'fourier_series_degree', 'empirical_value'))
    clocks['optimistic_value'] = np.array([optimistic_clock_value(om, M, ts, ys, ivars,)
                                           for om, M in zip(oms, Ms)])
    clocks['theoretical_value'] = np.array([theoretical_clock_value(om, M, ts, ys, ivars)
                                            for om, M in zip(oms, Ms)])

    # now filter and arrange the clocks
    good = (clocks['theoretical_value'] > MIN_THEORETICAL_VALUE) \
         & (clocks['theoretical_value'] > (np.max(clocks['theoretical_value']) / MAX_INTRASTAR_VALUE_RATIO))
    if np.sum(good) < 1:
        return None
    clocks = clocks[good]
    idx_sort = np.argsort(clocks['theoretical_value'])[::-1]
    clocks = clocks[idx_sort]
    idx_unique = np.logical_not(identify_resonances(clocks['angular_frequency'], np.pi * deltaf))
    clocks = clocks[idx_unique]
    good = (clocks['angular_frequency'] < (0.9999 * np.pi / deltat)) # magic nyquist?
    if np.sum(good) < 1:
        return None
    clocks = clocks[good]
    return clocks

'''database setup commands'''
def setup_db():
    with db.connect(CLOCKS_DB_FILE, timeout=120.0) as conn:
        with open("../sql/clocks_db_schema.sql", "r") as sql_file:
            sql_script = sql_file.read()
        conn.cursor().executescript(sql_script)
    load_star_table()
    load_dataset_table()
    load_task_table()

def get_db_connection():
    conn = db.connect(CLOCKS_DB_FILE, timeout=120.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def execute_query_and_close(query, retries = 5):
    try: 
        conn = get_db_connection()

        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(query)
        conn.commit()
        cursor.close()
        conn.close()
    except db.OperationalError as e:
        if retries > 0:
            print(f"OperationalError encountered. Retrying... ({retries} attempts left)")
            time.sleep(1)  # Wait a bit before retrying
            execute_query_and_close(query, retries - 1)
        else:
            print("Max retries reached. Could not execute query.")
            raise

def load_star_table():
    filename = "../data/KICids.csv"
    foo = Table.read(filename, format = "ascii.csv")
    if len(foo) > MAX_NUMBER_OF_STARS:
        foo = foo[:MAX_NUMBER_OF_STARS]
    kic_ids = [f"KIC{f:09d}" for f in foo['ID']]
    query = "DELETE FROM star;"
    execute_query_and_close(query)
    query = "INSERT INTO star (star_id) VALUES ('" + "'),('".join(kic_ids) + "');"
    execute_query_and_close(query)

def load_dataset_table():
    query = "DELETE FROM dataset"
    execute_query_and_close(query)
    query = """INSERT INTO dataset (dataset_id, description) 
    VALUES ('Kepler_long', 'Long-cadence data from the NASA Kepler Mission');"""
    execute_query_and_close(query)
    # query = """INSERT INTO dataset (dataset_id, description) 
    # VALUES ('Kepler_short', 'short-cadence data from the NASA Kepler Mission');"""
    # execute_query_and_close(query)

def load_task_table(select_kics = None):
    query = "DELETE FROM task;"
    execute_query_and_close(query)

    if select_kics is not None:
        if isinstance(select_kics, (list, tuple)):
            star_list = ','.join(f"'{sid}'" for sid in select_kics)
        else:
            # Allow direct string for flexibility
            star_list = f"'{select_kics}'"

        query = f"""
            INSERT INTO task(star_id, dataset_id)
            SELECT star.star_id, dataset.dataset_id
            FROM star
            CROSS JOIN dataset
            WHERE star.star_id IN ({star_list});
        """

    else:
        query = """INSERT into task(star_id, dataset_id) 
        SELECT star.star_id, dataset.dataset_id FROM star CROSS JOIN dataset;"""
    execute_query_and_close(query)
    count_query = "SELECT COUNT(*) FROM task;"
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(count_query)
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    print(f"Task table loaded with {count} entries.")

def update_message(star_id, dataset_id, message):
    query = f"""UPDATE task  
    SET message = "{message}" 
    WHERE star_id = "{star_id}" AND dataset_id = "{dataset_id}";"""
    execute_query_and_close(query)

def update_error_message(star_id, dataset_id, error_message):
    message = str(error_message).replace("'", "").replace("\n", " ")[:255]
    query = f"""
        UPDATE task
        SET error_message = '{message}'
        WHERE star_id = '{star_id}' AND dataset_id = '{dataset_id}';
    """
    print("clocks.update_error_message():", query)
    execute_query_and_close(query)

def start_one_task():
    conn = get_db_connection()
    conn.isolation_level = None  # autocommit off
    cursor = conn.cursor()
    
    try:
        cursor.execute("BEGIN IMMEDIATE")  # Lock immediately
        
        query1 = "SELECT star_id, dataset_id FROM task WHERE started IS NULL ORDER BY RANDOM() LIMIT 1;"
        cursor.execute(query1)
        foo = cursor.fetchall()
        
        if len(foo) == 0:
            conn.rollback()
            cursor.close()
            conn.close()
            print("clocks.start_one_task(): No unstarted tasks available.")
            sys.exit(0)

        star_id, dataset_id = foo[0]

        query2 = f"""UPDATE task SET 
        started = "{Time(Time.now(), format = "isot")}", 
        process_id = {os.getpid()} 
        WHERE star_id = "{star_id}" AND dataset_id = "{dataset_id}";"""
        cursor.execute(query2)

        conn.commit()
        cursor.close()
        conn.close()
        print(f"clocks.start_one_task() selected star_id={star_id}, dataset_id={dataset_id}")
        return star_id, dataset_id

    except Exception as e:
        conn.rollback()
        cursor.close()
        conn.close()
        raise

def output_clocks_to_db(star_id, dataset_id, clocks):
    """
    ## bugs:
    - This should take in a table, not a list of columns.
    - This duplicates code from connection and query code above, maybe?
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    for row in clocks:
        query = f"""
            INSERT INTO clock (star_id, dataset_id, angular_frequency, fourier_series_degree,
                               empirical_value, theoretical_value)
            VALUES ('{star_id}', '{dataset_id}', {row['angular_frequency']}, {row['fourier_series_degree']},
                    {row['empirical_value']}, {row['theoretical_value']});
        """
        cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()

def end_one_task(star_id, dataset_id):
    ##change this for when there is an error message
    #if error message is not null then do not set finished    
    query = f"""UPDATE task SET 
    finished = "{Time(Time.now(), format = "isot")}" 
    WHERE star_id = "{star_id}" AND dataset_id = "{dataset_id}";"""
    execute_query_and_close(query)

def restart_failed_tasks():
    """
    ##Bugs: 
    - The error_message query is way too universal
    """
    query = f"""
        UPDATE task
        SET finished = NULL,
        error_message = NULL,
        message = NULL
        WHERE error_message IS NOT NULL; 
        """
    execute_query_and_close(query)

    query = f"""
        UPDATE task
        SET started = NULL,
        process_id = NULL
        WHERE finished IS NULL
        AND started < "{Time(Time.now() - TimeDelta(600, format = "sec"), format = "isot")}"; 
        """
    execute_query_and_close(query)

def run_one_task():
    star_id, dataset_id = start_one_task()
    dataset_id = "Kepler_long"  #keep it lc for now
    clocks = best_clocks_in_star(star_id)
    if clocks is None:
        message = f"clocks.run_one_task(): No valid clocks found in {star_id}"
        print(message)
        update_message(star_id, dataset_id, message)
    else:
        message = f"clocks.run_one_task(): ---------> Found {len(clocks)} clocks in {star_id} with {(clocks['theoretical_value'])[0]:0.1e}"
        print(message)
        update_message(star_id, dataset_id, message)
        output_clocks_to_db(star_id, dataset_id, clocks)
    end_one_task(star_id, dataset_id)

def main():
    """ this is wrong"""
    if len(sys.argv) > 1 and sys.argv[1] == "db":
        setup_db()
        print("clocks.main(): Database setup complete.")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "schema":
        create_schema()
        print("clocks.main(): Database schema creation complete.")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        restart_failed_tasks()
        print("clocks.main(): Restarted all dead tasks.") 
        return

    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        time.sleep(5)
        while(True):
            run_one_task()
        return

    print("Usage:")
    print("  python clocks.py db        # to setup the database")
    print("  python clocks.py cleanup   # to restart failed tasks")
    print("  python clocks.py worker    # to run as a worker process")    
    return

if __name__ == "__main__":
    main()
