# Contributor Impact and Experience Analysis

## Project Overview

This project analyzes the relationship between contributor experience and their impact on software repositories, using the [Spring Boot](https://github.com/spring-projects/spring-boot) open source project as a case study. By examining Git commit history and contribution patterns, we aim to understand how developer experience correlates with productivity and code impact over time.

## Research Questions

1. How does contributor experience (measured in years) correlate with their impact on the codebase?
2. Who are the most impactful contributors over time?
3. What patterns emerge in contributor retention and productivity?
4. How does impact per year change as contributors gain more experience?
5. How can we accurately measure and improve software engineering efficiency in large-scale development teams?

## Data Sources

- Git repository commit history from [Spring Boot](https://github.com/spring-projects/spring-boot), an open source project with publicly available contribution data
- Contributor statistics including:
  - First and last contribution dates
  - Total commits, files changed, additions, and deletions
  - Years of experience in the repository
  - Active contribution periods
- Internal engineering metrics from platforms such as Jira (issue tracking), GitHub/GitLab (code commits, pull requests, and code review times), and CI/CD pipelines
- Industry benchmarks from public datasets like the GitHub Octoverse Report

### Data Collection

The data was collected using the following Git commands:

```bash
# Extract commit history with numstat information (additions/deletions per file)
git log --numstat --date=iso-strict --pretty=format:'---%n%H##%ad##%an##%ae' > data/git_numstat.csv

# Extract commit metadata (hash, date, author, email, subject)
git log --date=iso-strict --pretty=format:'%H##%ad##%an##%ae##%s' > data/git_commits.csv
```

## Analysis Techniques

1. **Contributor Experience Metrics**
   - Years since first commit
   - Active contribution periods
   - Consistency of contributions

2. **Impact Measurement**
   - Code volume metrics (additions, deletions, files changed)
   - Weighted impact scoring
   - Impact per year of experience

3. **Statistical Analysis**
   - Correlation between experience and impact
   - Regression analysis for predictive modeling
   - Cohort analysis of contributor groups
   - Descriptive and inferential statistics to identify trends and bottlenecks
   - Time-series analysis for productivity trends

4. **Visualization**
   - Time-series analysis of contribution patterns
   - Experience vs. impact scatter plots
   - Contributor retention curves
   - Stakeholder-ready dashboards

5. **Data Processing**
   - Data cleaning and integration from multiple systems

## Project Structure

- `data/` - Contains raw and processed Git data
  - `git_commits.csv` - Raw commit history
  - `git_numstat.csv` - Detailed statistics about each commit
  - `contributor_impact_dataset.csv` - Processed dataset with impact metrics

- `contributor_impact_analysis.ipynb` - Main analysis notebook examining overall contributor patterns
- `contributor_experience_vs_impact.ipynb` - Focused analysis on experience-impact relationship
- `plots/` - Generated visualizations and charts

## Key Insights

- Correlation between years of experience and code impact
- Identification of high-impact contributors
- Patterns in contributor retention and churn
- Effectiveness of experience as a predictor of productivity
- Ranked list of key factors that most influence engineering throughput and quality
- Actionable framework of engineering efficiency KPIs (e.g., lead time, cycle time, defect density)
- Recommendations for targeted process improvements, such as reducing review bottlenecks or optimizing sprint scope

## Why This Analysis Matters

Understanding the relationship between contributor experience and impact helps organizations:

- Better allocate resources and mentorship
- Identify potential knowledge gaps when experienced contributors leave
- Develop more effective onboarding strategies for new team members
- Create targeted retention programs for high-impact contributors

Software engineering is expensive, and inefficiency is even pricier. If we can pinpoint what slows teams down and what boosts their output, organizations can deliver features faster, improve quality, and reduce costs without simply hiring more people. Leaving this question unanswered means companies may waste time and money on "productivity" metrics that look appealing but don't actually help deliver better software.

By answering these questions, decision-makers will acquire a clear picture of where their teams truly excel, where they struggle, and which changes will deliver the most impact. This analysis provides data-driven insights that both technical and non-technical leaders can use to confidently implement improvements to team productivity and code quality.