/*
Question: What skills are required for the top-paying data scientist jobs?
- Use the top 1000 highest-paying data scientist jobs from first query
- Add the specific skills required for these roles
*/

WITH top_paying_skills AS (
    SELECT	
        job_id,
        job_title,
        salary_year_avg,
        name AS company_name
    FROM
        job_postings_fact
    LEFT JOIN company_dim ON job_postings_fact.company_id = company_dim.company_id
    WHERE
        job_title LIKE '%Data Scientist%'
        AND job_location LIKE '%United States%'
        AND salary_year_avg IS NOT NULL
    ORDER BY
        salary_year_avg DESC
    LIMIT 1000
)

SELECT 
    top_paying_skills.*,
    skills
FROM top_paying_skills
INNER JOIN skills_job_dim ON top_paying_skills.job_id = skills_job_dim.job_id
INNER JOIN skills_dim ON skills_job_dim.skill_id = skills_dim.skill_id
ORDER BY
    salary_year_avg DESC

