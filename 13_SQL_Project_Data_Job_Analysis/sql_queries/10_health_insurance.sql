/*
Is health insurance mentioned in higher-paying data science jobs?
- Analyze whether offering health insurance correlates with better pay.
*/

SELECT 
    job_health_insurance,
    COUNT(*) AS job_count,
    ROUND(AVG(salary_year_avg), 0) AS avg_salary
FROM job_postings_fact
WHERE 
    job_title_short = 'Data Scientist'
    AND salary_year_avg IS NOT NULL
    AND job_work_from_home = True
GROUP BY job_health_insurance
ORDER BY avg_salary DESC;
