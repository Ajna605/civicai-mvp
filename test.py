 {
    "id": "Q2",
    "question": "What is the population under 24 years in Coral Gables?",
    "category": "aggregation",
    "query": {
    "label": "Coral Gables",
    "measure": "population"
  },
    "expected": {
      "expected_numbers": [17233],
      "numeric_tolerance": 0
    },
    "known_issue": false
  },
  {
    "id": "Q2.1",
    "question": "What is the population over 50 in Coral Gables?",
    "category": "aggregation",
    "query": {
    "label": "Coral Gables",
    "measure": "population"
  },
    "expected": {
      "expected_numbers": [18992],
      "numeric_tolerance": 0
    },
    "known_issue": false
  },
  {
    "id": "Q2.2",
    "question": "How many people in Coral Gables are aged between 20 and 49?",
    "category": "aggregation",
    "query": {
    "label": "Coral Gables",
    "measure": "population"
  },
    "expected": {
      "expected_numbers": [17437],
      "numeric_tolerance": 0
    },
    "known_issue": false
  },
  {
    "id": "Q2.3",
    "question": "What percentage of the population is aged less than 24?",
    "category": "aggregation",
    "query": {
    "label": "Coral Gables",
    "measure": "percentage"
  },
    "expected": {
      "expected_numbers": [34.7],
      "numeric_type": "percent",
      "numeric_tolerance": 0
    },
    "known_issue": false
  },
  {
    "id": "Q2.4",
    "question": "What percentage of the population is aged over 50 years old?",
    "category": "aggregation",
    "query": {
    "label": "Coral Gables",
    "measure": "percentage"
  },
    "expected": {
      "expected_numbers": [38.0],
      "numeric_type": "percent",
      "numeric_tolerance": 0
    },
    "known_issue": false
  },
  {
    "id": "Q2.5",
    "question": "What percentage of the population is aged over 30 to 49 years old?",
    "category": "aggregation",
    "query": {
    "label": "Coral Gables",
    "measure": "percentage"
  },
    "expected": {
      "expected_numbers": [22.3],
      "numeric_type": "percent",
      "numeric_tolerance": 0
    },
    "known_issue": false
  },
  {
    "id": "Q3",
    "question": "What age range is most prominent in Coral Gables?",
    "category": "row_filter",
    "query": {
    "label": "Coral Gables",
    "measure": "max"
  },
    "expected": {
      "must_mention_all": ["15 to 19 years"]
    },
    "known_issue": false
  },
  {
    "id": "Q3.1",
    "question": "Which selected age categories which one is most populated?",
    "category": "row_filter",
    "query": {
    "label": "selected age categories",
    "measure": "max"
  },
    "expected": {
      "must_mention_all": ["16 years and over"]
    },
    "known_issue": false
  },
  {
    "id": "Q3.2",
    "question": "What is the most populated age range of males in Coral Gables?",
    "category": "row_filter",
    "query": {
    "label": "males",
    "measure": "max"
  },
    "expected": {
      "must_mention_all": ["15 to 19 years"]
    },
    "known_issue": false
  },
  {
    "id": "Q3.3",
    "question": "What is the most populated age range of women in Coral Gables?",
    "category": "row_filter",
    "query": {
    "label": "women",
    "measure": "max"
  },
    "expected": {
      "must_mention_all": ["15 to 19 years"]
        },
    "known_issue": false
  },
  {
    "id": "Q3.4",
    "question": "From the selected age categories of males, which one is highest?",
    "category": "row_filter",
    "query": {
    "label": "selected age categories",
    "measure": "max"
  },
    "expected": {
      "must_mention_all": ["16 years and over"]
    },
    "known_issue": false
  },
  {
    "id": "Q3.5",
    "question": "From the selected age categories of females, which one is highest?",
    "category": "row_filter",
    "query": {
    "label": "selected age categories",
    "measure": "max"
  },
    "expected": {
      "must_mention_all": ["16 years and over"]
    },
    "known_issue": false
  },
  {
    "id": "Q4",
    "question": "Create a bar chart of age groups vs total population.",
    "category": "chart_request",
    "query": {
    "label": "selected age categories",
    "measure": "max"
  },
    "expected": {
      "must_contain_labels": ["Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 19 years", "20 to 24 years" ],
      "min_points": 15
    },
    "known_issue": false
  },
  {
    "id": "Q4.1",
    "question": "Create a pie chart of male age groups over 50 years old in Coral Gables",
    "category": "chart_request",
    "expected": {
      "must_contain_labels": ["50 to 54 years", "55 to 59 years", "60 to 64 years", "65 to 69 years", "70 to 74 years"],
      "min_points": 5
    },
    "known_issue": false
  },
  {
    "id": "Q4.2",
    "question": "Create a bar chart of selected age categories from 5 to 44 years old.",
    "category": "chart_request",
    "expected": {
      "must_contain_labels": ["5 to 14 years", "15 to 17 years", "Under 18 years", "18 to 24 years", "15 to 44 years"],
      "min_points": 5
    },
    "known_issue": false
  },
  {
    "id": "Q4.3",
    "question": "Create a pie chart of the percentage of females in Coral Gables categorised by age groups",
    "category": "chart_request",
    "expected": {
      "must_contain_labels": ["Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 19 years"],
      "min_points": 15
    },
    "known_issue": false
  },
  {
    "id": "Q4.4",
    "question": "Create a pie chart of the population of females in Coral Gables categorised by age groups?",
    "category": "chart_request",
    "expected": {
      "must_contain_labels": ["Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 19 years"],
      "min_points": 15    
    },
    "known_issue": false
  },
  {
    "id": "Q4.5",
    "question": "Create a bar chart comparing the median age of males and females in Coral Gables",
    "category": "chart_request",
    "expected": {
      "must_contain_labels": ["Median age (years)"],
      "min_points": 2
    },
    "known_issue": false
  }
]