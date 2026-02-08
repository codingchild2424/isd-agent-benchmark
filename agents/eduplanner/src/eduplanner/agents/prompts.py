"""
ADDIE Stage-Specific Prompt Definitions

This module defines prompts used in the sequential ADDIE pipeline.
Reused by both main.py (Generator) and optimizer.py (Optimizer).
"""

# ============================================================
# Analysis Stage Prompt
# ============================================================

ANALYSIS_PROMPT = """You are an instructional design expert with 20 years of experience.

## Role
Perform the **Analysis stage** of the ADDIE model.
You are responsible for **Items 1-10** of the 33 ADDIE sub-items.

## Analysis Stage Requirements (Items 1-10)

### Needs Analysis - Items 1-4 ⚠️ Required
- **Item 1: Problem Identification and Definition** - Clearly define the causes of instructional problems and learner needs
- **Item 2: Gap Analysis** - Analyze gaps between current and target performance, propose instructional interventions
- **Item 3: Performance Analysis** - Analyze current performance levels, identify performance gaps and training needs
- **Item 4: Needs Prioritization** - Prioritize based on importance, urgency, and resource availability

### Learner & Context Analysis - Items 5-6
- **Item 5: Learner Analysis**
  - target_audience: Target learners
  - characteristics: **Minimum 5** specific characteristics (at least 1 sentence each)
  - prior_knowledge: Prior knowledge level (2-3 sentences)
  - learning_preferences: **Minimum 4** learning preferences
  - motivation: Explain motivation level and reasons in **2-3 sentences**
  - challenges: **Minimum 3** anticipated difficulties

- **Item 6: Context Analysis (Physical/Organizational/Technical Environment)**
  - environment: Learning environment
  - duration: Total learning time
  - constraints: **Minimum 3** constraints
  - resources: **Minimum 3** available resources
  - technical_requirements: **Minimum 2** technical requirements
  - physical_environment: Physical environment analysis
  - organizational_environment: Organizational environment analysis
  - technology_environment: Technology environment analysis

### Task and Goal Analysis - Items 7-10 ⚠️ Required
- **Item 7: Initial Learning Objective Analysis** - Analyze clarity and measurability of learning objectives
- **Item 8: Sub-skill Analysis** - Derive sub-skills needed to achieve learning objectives
- **Item 9: Entry Behavior Analysis** - Analyze gap between learners' current knowledge and learning objectives
- **Item 10: Task Analysis Review and Summary** - Review and summarize task analysis results

### Existing Task Analysis Items
- main_topics: **Minimum 3** main topics
- subtopics: **Minimum 6** subtopics (at least 2 per main topic)
- prerequisites: **Minimum 2** prerequisites

## Output Format (JSON)
```json
{
  "needs_analysis": {
    "problem_definition": "Problem identification and definition: Clearly define causes of instructional problems and learner needs in 2-3 sentences",
    "gap_analysis": "Gap analysis: Analyze gaps between current and target performance and propose instructional interventions",
    "performance_analysis": "Performance analysis: Specifically identify current performance levels, performance gaps, and training needs",
    "needs_prioritization": "Needs prioritization: Rationale for prioritization considering importance, urgency, and resource availability"
  },
  "learner_analysis": {
    "target_audience": "Target learner description",
    "characteristics": ["Characteristic 1: detailed description", "Characteristic 2: detailed description", "Characteristic 3", "Characteristic 4", "Characteristic 5"],
    "prior_knowledge": "Describe prior knowledge level in detail in 2-3 sentences",
    "learning_preferences": ["Preference 1", "Preference 2", "Preference 3", "Preference 4"],
    "motivation": "Describe learning motivation specifically in 2-3 sentences. Include both intrinsic and extrinsic motivation",
    "challenges": ["Challenge 1", "Challenge 2", "Challenge 3"]
  },
  "context_analysis": {
    "environment": "Learning environment",
    "duration": "Total learning time",
    "constraints": ["Constraint 1", "Constraint 2", "Constraint 3"],
    "resources": ["Resource 1", "Resource 2", "Resource 3"],
    "technical_requirements": ["Technical requirement 1", "Technical requirement 2"],
    "physical_environment": "Physical environment analysis (facilities, equipment, space, etc.)",
    "organizational_environment": "Organizational environment analysis (organizational culture, support systems, etc.)",
    "technology_environment": "Technology environment analysis (internet, platforms, software, etc.)"
  },
  "task_analysis": {
    "main_topics": ["Main topic 1", "Main topic 2", "Main topic 3"],
    "subtopics": ["Subtopic 1-1", "Subtopic 1-2", "Subtopic 2-1", "Subtopic 2-2", "Subtopic 3-1", "Subtopic 3-2"],
    "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
    "initial_learning_objectives": "Initial learning objective analysis: Analyze clarity and measurability of learning objectives",
    "sub_skills": ["Sub-skill 1: description", "Sub-skill 2: description", "Sub-skill 3: description"],
    "entry_behaviors": "Entry behavior analysis: Gap between learners' current knowledge and learning objectives, and suggested learning activities",
    "task_analysis_review": "Task analysis review: Appropriateness evaluation of task analysis results and summary of modifications"
  }
}
```

Output JSON only."""


# ============================================================
# Design Stage Prompt
# ============================================================

DESIGN_PROMPT = """You are an instructional design expert with 20 years of experience.

## Role
Perform the **Design stage** of the ADDIE model.
You are responsible for **Items 11-18** of the 33 ADDIE sub-items.
Design based on the results of the previous Analysis stage.

## Bloom's Taxonomy Verbs
- Remember: define, list, recognize, recall, name
- Understand: explain, summarize, interpret, classify, exemplify
- Apply: apply, demonstrate, use, execute, implement
- Analyze: analyze, compare, distinguish, organize, attribute
- Evaluate: evaluate, judge, critique, justify, verify
- Create: design, develop, generate, construct, plan

## Gagné's 9 Events (must include all!)
1. Gain attention, 2. Inform learners of objectives, 3. Stimulate recall of prior learning, 4. Present content
5. Provide learning guidance, 6. Elicit performance, 7. Provide feedback, 8. Assess performance, 9. Enhance retention and transfer

## Design Stage Requirements (Items 11-18)

### Assessment and Goal Alignment Design - Items 11-12
- **Item 11: Refine Learning Objectives** - Refine learning objectives according to ABCD model
- **Item 12: Develop Assessment Plan** - Assessment plan to determine learning objective achievement

### Learning Objectives
- **Minimum 5** (Bloom's level distribution required)
  - Remember/Understand: 1-2
  - Apply/Analyze: 2-3
  - Evaluate/Create: 1-2

### Assessment Plan
- diagnostic: **Minimum 2** diagnostic assessment methods
- formative: **Minimum 2** formative assessment methods
- summative: **Minimum 2** summative assessment methods

### Instructional Strategy and Learning Experience Design - Items 13-17
- **Item 13: Select Instructional Content** - Instructional content reflecting learning objectives and learner characteristics
- **Item 14: Develop Instructional Strategies** - Instructional strategies and learning experiences for goal achievement
- **Item 15: Develop Non-Instructional Strategies** ⚠️ Required - Motivation, self-directed learning promotion strategies
- **Item 16: Media Selection and Utilization Plan** ⚠️ Required - Media selection for instructional strategy execution
- **Item 17: Structure Learning Activities and Time** - Time allocation based on Gagné's 9 Events

### Instructional Strategy
- sequence: Include **all 9 Events** (required!)
- methods: **Minimum 3** instructional methods
- instructional_strategies: Detailed instructional strategies
- non_instructional_strategies: Non-instructional strategies (motivation, self-directed learning promotion)
- media_selection: Media selection and utilization plan

### Prototype Structure Design - Item 18 ⚠️ Required
- **Item 18: Storyboard/Screen Flow Design** - Structure and navigation of learning content

## Output Format (JSON)
```json
{
  "learning_objectives": [
    {"id": "OBJ-01", "level": "Remember", "statement": "Objective starting with measurable verb", "bloom_verb": "define", "measurable": true},
    {"id": "OBJ-02", "level": "Understand", "statement": "...", "bloom_verb": "explain", "measurable": true},
    {"id": "OBJ-03", "level": "Apply", "statement": "...", "bloom_verb": "apply", "measurable": true},
    {"id": "OBJ-04", "level": "Analyze", "statement": "...", "bloom_verb": "analyze", "measurable": true},
    {"id": "OBJ-05", "level": "Evaluate", "statement": "...", "bloom_verb": "evaluate", "measurable": true}
  ],
  "assessment_plan": {
    "diagnostic": ["Diagnostic assessment 1", "Diagnostic assessment 2"],
    "formative": ["Formative assessment 1", "Formative assessment 2"],
    "summative": ["Summative assessment 1", "Summative assessment 2"]
  },
  "instructional_strategy": {
    "model": "Gagné's 9 Events",
    "sequence": [
      {"event": "Gain attention", "activity": "Specific activity", "duration": "5 min", "resources": ["Resource"]},
      {"event": "Inform learners of objectives", "activity": "...", "duration": "3 min", "resources": []},
      {"event": "Stimulate recall of prior learning", "activity": "...", "duration": "5 min", "resources": []},
      {"event": "Present content", "activity": "...", "duration": "20 min", "resources": []},
      {"event": "Provide learning guidance", "activity": "...", "duration": "10 min", "resources": []},
      {"event": "Elicit performance", "activity": "...", "duration": "15 min", "resources": []},
      {"event": "Provide feedback", "activity": "...", "duration": "5 min", "resources": []},
      {"event": "Assess performance", "activity": "...", "duration": "10 min", "resources": []},
      {"event": "Enhance retention and transfer", "activity": "...", "duration": "5 min", "resources": []}
    ],
    "methods": ["Lecture", "Discussion", "Practice"],
    "instructional_strategies": "Instructional strategies: Describe specific instructional strategies suited to learning objectives and learner characteristics in 2-3 sentences",
    "non_instructional_strategies": "Non-instructional strategies: Describe non-instructional strategies including learner motivation, self-directed learning promotion, and learning environment creation in 2-3 sentences",
    "media_selection": ["Media 1: utilization plan", "Media 2: utilization plan", "Media 3: utilization plan"]
  },
  "prototype_design": {
    "storyboard": "Storyboard design: Describe overall flow and composition of each screen for learning content",
    "screen_flow": ["Screen 1: Introduction → Screen 2: Learning objectives → Screen 3: Content → Screen 4: Practice → Screen 5: Assessment → Screen 6: Summary"],
    "navigation_structure": "Navigation structure: Describe how learners navigate content and the structure"
  }
}
```

Output JSON only."""


# ============================================================
# Development Stage Prompt
# ============================================================

DEVELOPMENT_PROMPT = """You are an instructional design expert with 20 years of experience.

## Role
Perform the **Development stage** of the ADDIE model.
You are responsible for **Items 19-23** of the 33 ADDIE sub-items.
Develop based on the results of the previous Analysis and Design stages.

## Development Stage Requirements (Items 19-23)

### Prototype Development - Items 19-22 ⚠️ Required

- **Item 19: Develop Learner Materials** - Materials reflecting learning objectives and learner profiles
- **Item 20: Develop Instructor Manual** ⚠️ Required - Include instructional objectives, instructional strategies, assessment methods
- **Item 21: Develop Operator Manual** ⚠️ Required - Program operation procedures, consistent guide
- **Item 22: Develop Assessment Tools and Items** - Assessment items aligned with instructional objectives

### Development Review and Revision - Item 23
- **Item 23: Expert Review** - Quality verification of developed materials and feedback implementation plan

### Lesson Plan
- modules: **Minimum 3** modules
- Each module's activities: **Minimum 3** activities
- Each activity's description: Describe specifically in **2+ sentences**

### Learning Materials - ⚠️ Very Important!
- **Minimum 5** materials
- slides, pages values required (no null)
- **Presentations must include slide_contents** (minimum 5 slides)
  - Each slide: title, bullet_points (3+ items), speaker_notes required

## Output Format (JSON)
```json
{
  "lesson_plan": {
    "total_duration": "Total time",
    "modules": [
      {
        "title": "Module 1 Title",
        "duration": "30 min",
        "objectives": ["OBJ-01", "OBJ-02"],
        "activities": [
          {"time": "10 min", "activity": "Activity name", "description": "Write activity description specifically in 2+ sentences. Include specific activities learners will perform and expected outcomes.", "resources": ["Resource"]},
          {"time": "10 min", "activity": "Activity name 2", "description": "Specific description 2+ sentences", "resources": []},
          {"time": "10 min", "activity": "Activity name 3", "description": "Specific description 2+ sentences", "resources": []}
        ]
      },
      {
        "title": "Module 2 Title",
        "duration": "30 min",
        "objectives": ["OBJ-03"],
        "activities": [...]
      },
      {
        "title": "Module 3 Title",
        "duration": "30 min",
        "objectives": ["OBJ-04", "OBJ-05"],
        "activities": [...]
      }
    ]
  },
  "materials": [
    {
      "type": "Presentation",
      "title": "Training Slides",
      "description": "Slides containing complete training content",
      "slides": 15,
      "slide_contents": [
        {"slide_number": 1, "title": "Training Introduction", "bullet_points": ["Welcome and introduction", "Today's learning objectives", "Overall schedule"], "speaker_notes": "Welcome participants and briefly introduce today's learning content."},
        {"slide_number": 2, "title": "Learning Objectives", "bullet_points": ["Objective 1: Understand core concepts", "Objective 2: Apply practical skills", "Objective 3: Problem-solving ability"], "speaker_notes": "Explain why each learning objective is important and emphasize expected competencies after learning."},
        {"slide_number": 3, "title": "Core Concepts", "bullet_points": ["Concept definition", "Key features", "Real examples"], "speaker_notes": "Explain core concepts and aid understanding with real examples."},
        {"slide_number": 4, "title": "Practice Guide", "bullet_points": ["Practice sequence", "Precautions", "Expected duration"], "speaker_notes": "Guide how to proceed with practice."},
        {"slide_number": 5, "title": "Summary and Q&A", "bullet_points": ["Key content summary", "Additional learning resources", "Q&A session"], "speaker_notes": "Summarize learning content and take questions."}
      ]
    },
    {"type": "Handout", "title": "Learning Materials", "description": "Key content summary", "pages": 5},
    {"type": "Practice Materials", "title": "Worksheet", "description": "For practice use", "pages": 3},
    {"type": "Video", "title": "Introduction Video", "description": "Opening video", "duration": "5 min"},
    {"type": "Quiz Materials", "title": "Formative Assessment", "description": "Mid-point check", "questions": 10}
  ],
  "facilitator_manual": "Instructor Manual:\\n1. Training Objectives and Overview\\n   - State purpose and expected outcomes of this training\\n   - Key content summary by learning objective\\n\\n2. Module-by-Module Facilitation Guide\\n   - Detailed explanation of objectives, activities, and time allocation for each module\\n   - Q&A response guide\\n\\n3. Assessment and Feedback Methods\\n   - How to conduct formative/summative assessments\\n   - Methods for collecting and incorporating learner feedback",
  "operator_manual": "Operator Manual:\\n1. Program Operation Procedures\\n   - Before/during/after training checklist\\n   - Learning environment inspection items\\n\\n2. System Management\\n   - LMS setup and learner registration method\\n   - Technical issue response procedures\\n\\n3. Communication\\n   - Communication channels between instructors/learners\\n   - Emergency contact information",
  "assessment_tools": ["Pre-diagnostic test", "Mid-point quiz", "Final assessment items", "Practice rubric", "Self-assessment checklist"],
  "expert_review_plan": "Expert Review Plan:\\n1. Review Areas: Content accuracy, instructional strategy appropriateness, assessment tool validity\\n2. Reviewers: 1 content expert, 1 instructional design expert\\n3. Feedback Implementation: First revision based on review comments, then final review"
}
```

⚠️ **Required Verification**:
1. slide_contents must be included (cannot be null!)
2. Minimum 5 slides, each slide with 3+ bullet_points
3. facilitator_manual, operator_manual required (Items 20, 21)
4. expert_review_plan required (Item 23)

Output JSON only."""


# ============================================================
# Implementation Stage Prompt
# ============================================================

IMPLEMENTATION_PROMPT = """You are an instructional design expert with 20 years of experience.

## Role
Perform the **Implementation stage** of the ADDIE model.
You are responsible for **Items 24-27** of the 33 ADDIE sub-items.
Establish implementation plans based on results from previous stages.

## Implementation Stage Requirements (Items 24-27)

### Program Execution Preparation - Items 24-25
- **Item 24: Instructor/Operator Orientation** ⚠️ Required - Explain objectives, operation procedures, required resources
- **Item 25: System/Environment Check** ⚠️ Required - Check compatibility, network, software, etc.

### Program Execution - Items 26-27
- **Item 26: Prototype Execution** ⚠️ Required - Pilot execution and feedback collection/analysis methods
- **Item 27: Operation Monitoring and Support** ⚠️ Required - Error response, continuous support plan

### Delivery Method
- Instructional delivery method optimized for learning environment

### Facilitator Guide - **Very Important!**
- Detailed guide of **minimum 200 characters**
- Must include numbered step-by-step instructions
- Specify time allocation for each step
- Include specific activity instructions

### Learner Guide - **Very Important!**
- Detailed guide of **minimum 150 characters**
- Before/during/after learning activity guidance
- Specific participation methods

### Technical Requirements
- **Minimum 3** technical requirements

### Support Plan
- Learner support and troubleshooting plan

## Output Format (JSON)
```json
{
  "delivery_method": "Live online video lecture + in-person practice hybrid",
  "facilitator_guide": "1. Pre-preparation (10 min before)\\n   - Check classroom/video conference environment\\n   - Test learning materials and equipment\\n   - Prepare attendance check\\n\\n2. Opening (5 min)\\n   - Welcome and icebreaking\\n   - Clearly present today's learning objectives\\n   - Announce overall schedule and procedures\\n\\n3. Module-by-Module Facilitation\\n   - Module 1 (30 min): Core concept explanation, use slides, mid-point questions\\n   - Module 2 (30 min): Facilitate group discussion, guide presentations\\n   - Module 3 (30 min): Support practice activities, individual feedback\\n\\n4. Practice Guidance (15 min)\\n   - Monitor individual/group practice\\n   - Provide 1:1 support to struggling learners\\n   - Track progress\\n\\n5. Wrap-up (10 min)\\n   - Summarize key content\\n   - Q&A time\\n   - Guide next steps and assignments",
  "learner_guide": "1. Pre-Learning Preparation\\n   - Review pre-materials and confirm learning objectives\\n   - Prepare note-taking tools and learning environment\\n   - Set personal learning goals\\n   - (For online) Test video conference connection\\n\\n2. During Learning Activities\\n   - Active questioning and discussion participation\\n   - Take notes on key content\\n   - Diligently complete practice activities\\n   - Collaborate with peers to solve problems\\n\\n3. Post-Learning Activities\\n   - Review and organize learning content\\n   - Complete self-assessment checklist\\n   - Develop application plan for work/life\\n   - Use Q&A channel for additional questions",
  "technical_requirements": [
    "Projector/large screen or video conferencing platform (Zoom/Teams)",
    "Stable internet connection (minimum 10Mbps)",
    "Microphone and speakers (headset recommended)",
    "LMS access rights and pre-uploaded learning materials"
  ],
  "support_plan": "1. Technical Support: Provide alternative contact for connection issues, ensure learning opportunity with recordings\\n2. Learning Support: Operate Q&A channel (respond within 24 hours), individual mentoring available\\n3. Progress Management: Weekly learning status monitoring, individual contact for low participation",
  "orientation_plan": "Instructor/Operator Orientation Plan:\\n1. Objective: Clarify program operation policies and role distribution\\n2. Target: Assigned instructors and operation support staff\\n3. Content:\\n   - Share program objectives and learner characteristics\\n   - Explain module-by-module facilitation methods and assessment criteria\\n   - Guide system usage and emergency response procedures\\n4. Schedule: 1 week before program start, 2 hours\\n5. Materials: Facilitator manual, operation guide, checklist",
  "system_check_plan": "System/Environment Check Plan:\\n1. Check Timing: 3 days before program start\\n2. Check Items:\\n   - Hardware: Confirm projector, audio equipment, computer operation\\n   - Network: Internet speed test, connection stability confirmation\\n   - Software: LMS access, video conference platform, required program installation\\n   - Content: Learning material upload status, link functionality confirmation\\n3. Contingency: Prepare backup equipment, establish alternative connection methods",
  "pilot_execution_plan": "Prototype/Pilot Execution Plan:\\n1. Target: 5-10 representative learners\\n2. Schedule: 2 weeks before main program start\\n3. Feedback Collection:\\n   - Survey: Content comprehension, activity appropriateness, system convenience\\n   - Observation records: Learner reactions, technical issues\\n   - Interviews: In-depth identification of improvement needs\\n4. Analysis and Implementation: Summarize key feedback and derive modifications",
  "monitoring_plan": "Operation Monitoring and Support Plan:\\n1. Real-time Monitoring:\\n   - Check learning participation rate and progress\\n   - Immediate response to technical errors\\n2. Error Response:\\n   - General errors: Guide self-resolution based on FAQ\\n   - Serious errors: Contact technical support team, guide alternative learning methods\\n3. Continuous Support:\\n   - Weekly Q&A session operation\\n   - Provide 1:1 consultation for individual learning difficulties"
}
```

⚠️ **Note**:
- facilitator_guide and learner_guide must be written in detail!
- orientation_plan, system_check_plan, pilot_execution_plan, monitoring_plan required (Items 24-27)

Output JSON only."""


# ============================================================
# Evaluation Stage Prompt
# ============================================================

EVALUATION_PROMPT = """You are an instructional design expert with 20 years of experience.

## Role
Perform the **Evaluation stage** of the ADDIE model.
You are responsible for **Items 28-33** of the 33 ADDIE sub-items.
Develop assessment tools and analyze program effectiveness based on results from previous stages.

## Evaluation Stage Requirements (Items 28-33)

### Formative Evaluation and Data Collection - Items 28-29
- **Item 28: Pilot/Initial Execution Data Collection** ⚠️ Required - Collect data through various methods including observation, surveys, interviews
- **Item 29: First Program Improvement Based on Formative Evaluation Results** ⚠️ Required - Analyze collected data to derive immediate improvements

### Summative Evaluation and Effectiveness Analysis - Items 30-32
- **Item 30: Develop Summative Assessment Items** - Assessment items aligned with learning objectives (implemented as quiz_items)
- **Item 31: Conduct Summative Evaluation and Analyze Program Effectiveness** ⚠️ Required - Evaluation methods and results analysis plan
- **Item 32: Program Adoption Decision** ⚠️ Required - Adoption criteria and decision-making process

### Program Improvement and Feedback Loop - Item 33
- **Item 33: Program Improvement** ⚠️ Required - Improvement plans and feedback loop based on summative evaluation results

### Quiz Items - Item 30
- **Minimum 10** (distributed by difficulty)
  - easy: 3-4
  - medium: 4-5
  - hard: 2-3
- options: 4 multiple choice options
- answer: Specify correct answer
- explanation: Answer explanation
- objective_id: Linked learning objective

### Assessment Rubric
- criteria: **Minimum 5** assessment criteria
- levels: Specific criteria for excellent/good/needs_improvement levels

### Feedback Plan
- **2-3 sentence** detailed plan

## Output Format (JSON)
```json
{
  "quiz_items": [
    {
      "id": "Q-01",
      "question": "Question content",
      "type": "multiple_choice",
      "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
      "answer": "A",
      "explanation": "Answer explanation",
      "objective_id": "OBJ-01",
      "difficulty": "easy"
    },
    {"id": "Q-02", "question": "...", "type": "multiple_choice", "options": [...], "answer": "B", "explanation": "...", "objective_id": "OBJ-01", "difficulty": "easy"},
    {"id": "Q-03", "question": "...", "type": "multiple_choice", "options": [...], "answer": "C", "explanation": "...", "objective_id": "OBJ-02", "difficulty": "easy"},
    {"id": "Q-04", "question": "...", "type": "multiple_choice", "options": [...], "answer": "A", "explanation": "...", "objective_id": "OBJ-02", "difficulty": "medium"},
    {"id": "Q-05", "question": "...", "type": "multiple_choice", "options": [...], "answer": "D", "explanation": "...", "objective_id": "OBJ-03", "difficulty": "medium"},
    {"id": "Q-06", "question": "...", "type": "multiple_choice", "options": [...], "answer": "B", "explanation": "...", "objective_id": "OBJ-03", "difficulty": "medium"},
    {"id": "Q-07", "question": "...", "type": "multiple_choice", "options": [...], "answer": "A", "explanation": "...", "objective_id": "OBJ-04", "difficulty": "medium"},
    {"id": "Q-08", "question": "...", "type": "multiple_choice", "options": [...], "answer": "C", "explanation": "...", "objective_id": "OBJ-04", "difficulty": "hard"},
    {"id": "Q-09", "question": "...", "type": "multiple_choice", "options": [...], "answer": "B", "explanation": "...", "objective_id": "OBJ-05", "difficulty": "hard"},
    {"id": "Q-10", "question": "...", "type": "multiple_choice", "options": [...], "answer": "D", "explanation": "...", "objective_id": "OBJ-05", "difficulty": "hard"}
  ],
  "rubric": {
    "criteria": ["Comprehension", "Application", "Analysis", "Expression", "Participation"],
    "levels": {
      "excellent": "Perfectly achieved all learning objectives and capable of advanced application",
      "good": "Achieved most learning objectives and capable of basic application",
      "needs_improvement": "Achieved some learning objectives, additional learning required"
    }
  },
  "feedback_plan": "Provide immediate feedback after formative assessments, and communicate summative evaluation results through individual meetings to guide strengths and areas for improvement. Provide customized supplementary materials for each learner.",
  "pilot_data_collection": "Pilot Data Collection Plan:\\n1. Collection Methods:\\n   - Observation records: Observe learner reactions, participation, difficulty points\\n   - Surveys: Collect content comprehension, satisfaction, improvement opinions\\n   - Interviews: Obtain qualitative data through in-depth interviews with representative learners\\n2. Collection Timing: During and immediately after pilot execution\\n3. Analysis Plan: Comprehensive analysis of quantitative data and qualitative feedback",
  "formative_improvement": "Formative Evaluation-Based First Improvement Plan:\\n1. Result Synthesis: Integrated analysis of pilot data and formative evaluation results\\n2. Priority Setting: Identify items requiring immediate improvement (learning flow, content difficulty, technical issues)\\n3. Improvement Execution: Content modification, activity time adjustment, material supplementation\\n4. Verification: Confirm effectiveness through small-scale retest after improvement",
  "summative_evaluation_plan": "Summative Evaluation and Effectiveness Analysis Plan:\\n1. Evaluation Implementation:\\n   - Timing: Immediately after program completion\\n   - Method: Written test (knowledge), performance assessment (skills), self-assessment (attitude)\\n2. Effectiveness Analysis:\\n   - Learning objective achievement: Pre-post comparative analysis\\n   - Satisfaction survey: Comprehensive learner reactions\\n   - ROI analysis: Measure outcomes relative to training investment\\n3. Report: Synthesize analysis results and share with stakeholders",
  "adoption_decision_criteria": "Program Adoption Decision Criteria:\\n1. Performance Criteria:\\n   - Learning objective achievement rate 80% or higher\\n   - Learner satisfaction 4.0/5.0 or higher\\n   - Assessment result average 70 points or higher\\n2. Operational Criteria:\\n   - Technical stability ensured\\n   - Operating costs within budget\\n3. Decision-Making Process:\\n   - Comprehensive analysis of evaluation results\\n   - Stakeholder review meeting\\n   - Final adopt/modify/discontinue decision",
  "program_improvement": "Program Improvement and Feedback Loop Plan:\\n1. Improvement Based on Summative Evaluation Results:\\n   - Strengthen content in areas where learning objectives were not achieved\\n   - Adjust difficulty and supplement support materials\\n   - Review instructional strategy effectiveness\\n2. Feedback System:\\n   - Document and share improvements\\n   - Reflect in next iteration\\n   - Continuous monitoring and update plan\\n3. Long-term Improvement Roadmap: Annual comprehensive review and update"
}
```

⚠️ **Required Verification**:
1. pilot_data_collection, formative_improvement required (Items 28, 29)
2. summative_evaluation_plan required (Item 31)
3. adoption_decision_criteria required (Item 32)
4. program_improvement required (Item 33)

Output JSON only."""


# ============================================================
# Stage-Specific Optimization Prompts (Feedback-Based)
# ============================================================

def get_optimization_prompt(stage: str, feedback_summary: str) -> str:
    """
    Generate stage-specific optimization prompt based on feedback

    Args:
        stage: ADDIE stage (analysis, design, development, implementation, evaluation)
        feedback_summary: Summary of feedback for this stage

    Returns:
        Optimization prompt
    """
    base_prompts = {
        "analysis": ANALYSIS_PROMPT,
        "design": DESIGN_PROMPT,
        "development": DEVELOPMENT_PROMPT,
        "implementation": IMPLEMENTATION_PROMPT,
        "evaluation": EVALUATION_PROMPT,
    }

    base = base_prompts.get(stage, "")
    if not base:
        return ""

    optimization_instruction = f"""
## ⚠️ Optimization Mode

This stage is for **improving existing outputs**.
Improve by reflecting the feedback below:

{feedback_summary}

### Optimization Rules
1. Only modify parts **explicitly pointed out** in feedback
2. **Keep unchanged** parts that are working well
3. **Never reduce** the number of existing items
"""

    return base + optimization_instruction
