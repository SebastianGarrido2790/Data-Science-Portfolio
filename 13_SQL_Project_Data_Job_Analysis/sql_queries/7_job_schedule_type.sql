/*
How does job schedule type impact salary for remote data scientist roles?
- Investigate how job types (e.g., full-time, contract) influence salaries.
*/


SELECT 
    job_schedule_type,
    COUNT(*) AS count_jobs,
    ROUND(AVG(salary_year_avg), 0) AS avg_salary
FROM job_postings_fact
WHERE 
    job_title_short = 'Data Scientist'
    AND salary_year_avg IS NOT NULL
    AND job_work_from_home = True
GROUP BY job_schedule_type
ORDER BY avg_salary DESC;
