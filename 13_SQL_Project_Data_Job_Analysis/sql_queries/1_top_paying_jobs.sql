/*
Question: What are the top-paying data science jobs?
- Identify the top 30 highest-paying data science roles that are available remotely in USA
- Focuses on job postings with specified salaries (remove nulls)
- Include company names
*/

SELECT 
    job_id,
    job_title,
    job_location,
    job_posted_date,
    salary_year_avg,
    name AS company_name
FROM
    job_postings_fact
LEFT JOIN
    company_dim ON job_postings_fact.company_id = company_dim.company_id
WHERE
    job_title LIKE '%Data Scientist%'
    AND job_location LIKE '%United States%'
    AND salary_year_avg IS NOT NULL
ORDER BY
    salary_year_avg DESC
LIMIT 30;