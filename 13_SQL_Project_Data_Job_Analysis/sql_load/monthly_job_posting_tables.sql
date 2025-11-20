-- Create tables for job postings for each month of the year

-- January
CREATE TABLE january_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 1;

-- February
CREATE TABLE february_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 2;

-- March
CREATE TABLE march_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 3;

-- April
CREATE TABLE april_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 4;

-- May
CREATE TABLE may_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 5;

-- June
CREATE TABLE june_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 6;

-- July
CREATE TABLE july_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 7;

-- August
CREATE TABLE august_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 8;

-- September
CREATE TABLE september_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 9;

-- October
CREATE TABLE october_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 10;

-- November
CREATE TABLE november_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 11;

-- December
CREATE TABLE december_jobs AS
SELECT *
FROM job_postings_fact
WHERE EXTRACT(MONTH FROM job_posted_date) = 12;

-- If you're analyzing rather than storing, use views or CTEs for a more lightweight approach.
-- Views or CTEs are much more efficient and cleaner than creating physical tables.

-- Dynamically create monthly views instead of physical tables
DO $$
DECLARE
    month_num INT;
    month_name TEXT;
    month_names TEXT[] := ARRAY[
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ];
BEGIN
    FOR month_num IN 1..12 LOOP
        month_name := month_names[month_num];

        -- Drop the view if it already exists to avoid errors
        EXECUTE format('DROP VIEW IF EXISTS %I_jobs_view;', month_name);

        -- Create a view for each month filtering on job_posted_date
        EXECUTE format('
            CREATE VIEW %I_jobs_view AS
            SELECT *
            FROM job_postings_fact
            WHERE EXTRACT(MONTH FROM job_posted_date) = %s;',
            month_name, month_num
        );
    END LOOP;
END $$;

-- No data is duplicated, and the views stay in sync with the original job_postings_fact table.
-- Once created:

SELECT * FROM july_jobs_view WHERE job_country = 'United States';
