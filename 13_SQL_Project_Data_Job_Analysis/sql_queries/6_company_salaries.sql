/*
Which companies offer the highest average salaries for remote data scientists?
- Highlight companies consistently paying higher salaries for remote data science positions.
*/

SELECT 
    company_dim.name AS company_name,
    COUNT(job_postings_fact.job_id) AS num_jobs,
    ROUND(AVG(salary_year_avg), 0) AS avg_salary
FROM job_postings_fact
INNER JOIN company_dim ON job_postings_fact.company_id = company_dim.company_id
WHERE 
    job_title_short = 'Data Scientist'
    AND salary_year_avg IS NOT NULL
    AND job_work_from_home = True
GROUP BY company_dim.name
HAVING COUNT(job_postings_fact.job_id) >= 3
ORDER BY avg_salary DESC
LIMIT 15;
