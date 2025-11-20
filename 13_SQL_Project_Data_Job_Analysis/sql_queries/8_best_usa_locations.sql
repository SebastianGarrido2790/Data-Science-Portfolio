/*
Which U.S. locations offer the best pay for remote data scientists?
- Explore regional differences in salary even within remote roles.
*/

SELECT 
    search_location,
    COUNT(*) AS job_count,
    ROUND(AVG(salary_year_avg), 0) AS avg_salary
FROM job_postings_fact
WHERE 
    job_title_short = 'Data Scientist'
    AND salary_year_avg IS NOT NULL
    AND job_work_from_home = True
    AND search_location LIKE '%United States%'
GROUP BY search_location
HAVING COUNT(*) >= 5
ORDER BY avg_salary DESC
LIMIT 15;
