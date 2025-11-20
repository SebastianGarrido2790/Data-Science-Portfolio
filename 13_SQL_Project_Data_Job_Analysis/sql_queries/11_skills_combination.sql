/*
Skill combinations most associated with high-paying roles
- This one uses a skill pair frequency strategy and is more advanced.
- Identify popular skill pairings in remote, high-paying jobs to guide curriculum design or upskilling.
*/

WITH skill_pairs AS (
    SELECT 
        sj1.job_id,
        LEAST(sj1.skill_id, sj2.skill_id) AS skill_1,
        GREATEST(sj1.skill_id, sj2.skill_id) AS skill_2
    FROM skills_job_dim sj1
    JOIN skills_job_dim sj2 
        ON sj1.job_id = sj2.job_id AND sj1.skill_id < sj2.skill_id
    JOIN job_postings_fact jp ON jp.job_id = sj1.job_id
    WHERE 
        jp.job_title_short = 'Data Scientist'
        AND jp.salary_year_avg IS NOT NULL
        AND jp.job_work_from_home = True
)

SELECT 
    sd1.skills AS skill_a,
    sd2.skills AS skill_b,
    COUNT(*) AS pair_count
FROM skill_pairs
JOIN skills_dim sd1 ON sd1.skill_id = skill_pairs.skill_1
JOIN skills_dim sd2 ON sd2.skill_id = skill_pairs.skill_2
GROUP BY skill_a, skill_b
HAVING COUNT(*) >= 5
ORDER BY pair_count DESC
LIMIT 15;
