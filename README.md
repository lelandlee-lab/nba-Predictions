
Predicting NBA Game Winners

Authors: Leland Lee (@lelandlee-lab) and Haitham (haithamassaf)

Description and Research Topic:
The goal of our project is to use machine learning models to predict major NBA awards and end-of-season teams. Rather than building a normal supervised classifier, we focus on uncovering underlying structure in player performance data using techniques such as K-Means clustering, StandardScaler, and custom scoring formulas. Through this approach, we identify offensive and defensive tiers, evaluate overall star levels, and generate accurate predictions for MVP, DPOY, and Sixth Man awards. We also construct All-NBA First, Second, and Third Teams, All-Defensive Teams, and All-Star selections by conference. Our emphasis is on model interpretation, data visualization, and incorporating machine learning to basketball analytics.

Project outline:

1. Week 1: Set up GitHub repository, specify roles, outline project order, gather initial player statistics.
2. Week 2: Collect and organize player-level data. Prepare dataset and cerate features
3. Week 3: Apply StandardScaler and implement K-Means clustering in order to create offensive and defensive tiers.
4. Week 4: Build custom scoring formulas and create award predictions (MVP, DPOY, Sixth Man). Begin visualization for our data.
5. Week 5: Take notes on preformance and make a detailed report and finalize presentation.

Data Collection Plan:

Leland Lee:
1. Collect team stats like rebounds, turnovers, points, etc.
2. Engineer features for clustering such as usage rate, ratings, and role indicators.


Haitham:
1. Gather advanced metrics including ORTG, DRTG, net rating, and in game/off game stats.
2.  Solidify and identify lineup roles
3. Provide supporting datasets and ensure data is acceptable.


Model Plans:

Leland Lee:
1. Assist in testing different clustering approaches and verifying award predictions.
2. Support feature selection and metric research.
3. Help with visualization and analysis 


Haitham:
1. Implement StandardScaler and K-Means clustering to generate offensive and defensive tiers.
2. Designed award scoring formulas for MVP, DPOY, and Sixth Man using engineered features and cluster information.
3. Helped with visualization and analysis


Project Timeline
       
Week	Dates	          Updated Goals
1	    Oct 27 – Nov 2	Set up repo, assign roles, begin collecting player stats
2	    Nov 3 – Nov 9	  Clean player dataset, work on offensive/defensive features
3	    Nov 10 – Nov 16	Apply StandardScaler + K-Means; interpret clusters
4	    Nov 17 – Nov 23	Build award scoring system; generate trial predictions
5	    Nov 24 – Nov 30	Finalize award results, create All-NBA/All-Defensive/All-Star teams, write analysis, prepare slides

Repository Structure
├── Week 14/                               
│   ├── .ipynb_checkpoints/
│   └── PoW14-Updated timeline- Predicting NBA Games.ipynb
│
├── Analysis and Virtualization.ipynb    
├── Data Processing.ipynb                 
├── Model Construction.ipynb               
│
├── database_24_25.csv                   
├── README.md                             
├── LICENSE.txt                           
├── .gitignore                  

Installation Instructions

- Clone Repository: git clone https://github.com/lelandlee-lab/nba-Predictions.git
                    cd nba-Predictions

- Install Dependencies: pip install pandas numpy scikit-learn matplotlib seaborn

How to Run Project?
1. Run Data Processing.ipynb
- This notebook loads the raw player dataset, cleans and structures the stats, and creates the offensive and defensive features used throughout the project. It then produces a standardized version of the dataset that is ready for clustering.
2. Run Model Construction.ipynb\
  - This notebook applies StandardScaler to normalize all numerical features and then runs K-Means clustering to create offensive and defensive player tiers. It also outputs the resulting cluster labels  that will later be to determine the award scoring.
3. Run Analysis and Visualization.ipynb
- This notebook calculates scoring formulas for MVP, DPOY, and Sixth Man candidates using both raw statistics and cluster information. It also generates All-NBA Teams, All-Defensive Teams, and All-Star selections. Visualizations such as scatterplot and bar charts are created to interpret the results, and a final outline of insights is provided.

Future Improvements
- Incorporating more advanced stats in order to make predictions more accurate
- Trying other clustering models to see if those produce more accurate predictions
- Expanding our dataset in order to see if it can predict rookie of the year and most impproved player of the year awards
- Making a interactive UI so users can explore player tiers and see award predicitons through an app or website
- Expand dataset using previous seasons instead of being limited to current year stats

Conclusion
- This project shows us how unsupervised learning can reveal a meaningful outline of what NBA awards can look like. By inocroporating different features like clustering, visualization, and many statistical patterns we can see that it is possible to align these models with real world perceptions of these players.


