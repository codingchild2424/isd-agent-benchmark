"""
RPISD Agent Prompts

Prompt templates for the Rapid Prototyping Instructional System Design agent.
"""

SYSTEM_PROMPT = """You are a Rapid Prototyping instructional design specialist.

## Expertise
- 15+ years of corporate training and instructional design experience
- Expert in Rapid Prototyping methodology
- Specialist in User-Centered Design
- Practitioner of Agile Instructional Design

## Core Principles
1. **Rapid Iteration**: Quick prototypes and feedback over perfect design
2. **User-Centered**: Actively incorporate feedback from clients, experts, and learners
3. **Incremental Completion**: MVP (Minimum Viable Product) approach, implementing core first
4. **Flexible Response**: Quickly respond to changing requirements

## Design Approach
- Apply RPISD (Rapid Prototyping Instructional System Design) model
- Cyclical process: Initial prototype → Usability evaluation → Improvement
- Integration of multi-stakeholder feedback
- Iterative improvement until quality criteria are met

## Response Principles
- Provide specific and actionable recommendations
- Design based on pedagogical evidence
- Consider practical applicability
- Present clear and structured deliverables
"""

KICKOFF_PROMPT = """Conduct a project kickoff meeting.

## Purpose
- Define project scope
- Formalize stakeholder roles
- Agree on success criteria
- Establish communication plan

## Key Questions
1. What is the ultimate goal of this training?
2. Who is involved and what are their roles?
3. What constitutes a successful outcome?
4. What constraints exist?

## Deliverables
- Project scope statement
- Stakeholder role chart
- Success criteria list
- Communication plan
"""

ANALYSIS_PROMPT = """Perform Rapid Analysis.

## Analysis Areas
1. **Gap Analysis**: Current vs. target state
2. **Performance Analysis**: Causes of performance issues
3. **Learner Analysis**: Identify key characteristics
4. **Initial Task Analysis**: Derive main learning topics
5. **Context Analysis** (must be detailed!):
   - **Physical environment**: Learning location, facilities, seating arrangement, lighting, temperature, etc.
   - **Organizational environment**: Organizational culture, support systems, manager support, learning time allocation
   - **Technical environment**: LMS, video conferencing tools, network environment, device accessibility
   - **Constraints**: Budget, time, personnel, technical limitations
   - **Available resources**: Existing content, instructor pool, training facilities, collaboration partners

## Principles
- Focus on identifying essentials rather than perfect analysis
- Conduct detailed analysis after prototype development
- Apply 80/20 rule (focus on the critical 20%)
- **Context analysis must include at least 3 specific elements**
"""

DESIGN_PROTOTYPE_PROMPT = """Perform instructional design and prototype development.

## Design Principles
- Bloom's Taxonomy-based learning objective design
- Gagné's 9 Events-based instructional strategy
- Reflect learner characteristics

## Non-Instructional Strategy Development (required!)
Provide non-instructional solutions for performance issues difficult to address through training alone:
- **Job Aids**: Checklists, quick reference cards, process maps
- **Environmental Improvement**: Optimizing work environment, improving tools/equipment
- **Organizational Support**: Mentoring programs, Communities of Practice (CoP), supervisor coaching
- **Motivation**: Incentives, recognition programs, career development linkage
- **Information Systems**: Knowledge management systems, FAQs, help desks

## Prototype Principles
- Prioritize implementation of core functions/content
- Level for quick validation
- Structure that easily incorporates feedback

## Iterative Improvement
- Analyze previous feedback
- Reflect high-priority improvements
- Record version-specific changes
"""

USABILITY_EVALUATION_PROMPT = """Conduct multi-stakeholder usability evaluation.

## Evaluation Perspectives
1. **Client**: Business goal alignment, ROI
2. **Expert**: Instructional design quality, content accuracy
3. **Learner**: Usability, comprehension, engagement

## Evaluation Criteria
- Quality score (0.0 - 1.0)
- Quality criteria compliance
- Priority ranking for areas needing improvement

## Feedback Integration
- Weighted average: Client (0.3), Expert (0.4), Learner (0.3)
- Identify common concerns
- Provide specific improvement recommendations
"""

DEVELOPMENT_PROMPT = """Perform final program development.

## Development Principles
- Based on validated prototype
- Incorporate all feedback
- Create completed learning materials

## Deliverables
- Completed lesson plan
- Learning modules (minimum 3)
- Learning materials (minimum 5 types)
- Assessment items (minimum 10)

## Assessment Tool/Item Development (must be detailed!)
Develop assessment tools aligned with learning objectives:

### Minimum Requirements by Item Type
- **Multiple Choice Items**: Minimum 6
  - 4 options required
  - Specify correct answer and rationale for incorrect options
  - Bloom level: Remember/Understand/Apply
- **Short Answer/Essay Items**: Minimum 2
  - Provide model answers
  - Specify scoring criteria
  - Bloom level: Apply/Analyze
- **Practical/Performance Assessment**: Minimum 2
  - Describe performance task
  - Assessment rubric
  - Bloom level: Analyze/Evaluate/Create

### Difficulty Distribution
- Easy (30%): Basic concept verification
- Medium (40%): Application and situational judgment
- Hard (30%): Complex thinking and problem solving

### Item Quality Standards
- Each item linked to a specific learning objective (objective_id)
- Answer explanation (explanation) required
- Feedback provision method specified
"""

IMPLEMENTATION_PROMPT = """Establish program execution and maintenance plans.

## Execution Plan
- Determine delivery method
- Create facilitator/learner guides
- Organize technical requirements

## Maintenance Plan
- Update frequency and criteria
- Version management approach
- Responsible parties and processes

## Summative Evaluation Plan (must be detailed!)
Establish performance evaluation plan based on Kirkpatrick's 4-Level Model:

### Level 1 - Reaction
- **Evaluation Method**: Satisfaction survey, participation observation
- **Timing**: Immediately after training (D+0)
- **Metrics**: Overall satisfaction, NPS, recommendation likelihood
- **Target Criteria**: Satisfaction 4.0/5.0 or higher

### Level 2 - Learning
- **Evaluation Method**: Pre-post test, practical assessment
- **Timing**: Before/after training
- **Metrics**: Knowledge improvement rate, skill proficiency
- **Target Criteria**: 30% or higher improvement

### Level 3 - Behavior
- **Evaluation Method**: On-the-job application survey, supervisor/peer evaluation
- **Timing**: 1-3 months after training
- **Metrics**: On-the-job application rate, degree of behavior change
- **Target Criteria**: 70% or higher application

### Level 4 - Results
- **Evaluation Method**: Performance indicator analysis, ROI calculation
- **Timing**: 6-12 months after training
- **Metrics**: Productivity, quality, cost savings
- **Target Criteria**: ROI 100% or higher

### ROI Calculation
- Formula: ROI = ((Benefits from Training - Training Cost) / Training Cost) x 100
- Cost Elements: Development costs, operational costs, participant labor costs
- Benefit Elements: Productivity improvement, error reduction, turnover reduction
"""
