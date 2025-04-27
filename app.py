from flask import Flask, request, render_template
import random
import heapq
import pandas as pd

app = Flask(__name__)

# Load feedback dataset (for rule-based NLP)
feedback_df = pd.read_csv("data/feedback.csv")

# Task class for scheduling
class Task:
    def __init__(self, name, topic, duration, deadline, difficulty):
        self.name = name
        self.topic = topic
        self.duration = duration  # hours
        self.deadline = deadline  # days
        self.difficulty = difficulty  # 1-10

# Simplified NLP: Rule-based sentiment and topic detection
def process_feedback(feedback):
    feedback = feedback.lower()
    sentiment = "positive"
    if any(word in feedback for word in ["hard", "difficult", "stuck", "struggling", "tough", "weak", "overwhelmed"]):
        sentiment = "negative"
    topic = "QA"
    if "dilr" in feedback:
        topic = "DILR"
    elif "varc" in feedback:
        topic = "VARC"
    return sentiment, topic

# A* Scheduling
def a_star_schedule(tasks, proficiency):
    def heuristic(state):
        return sum(t.duration / proficiency[t.topic] for t in state['remaining'])

    start = {'schedule': [], 'remaining': tasks, 'time': 0}
    queue = [(0, start)]
    visited = set()

    while queue:
        cost, state = heapq.heappop(queue)
        if not state['remaining']:
            return state['schedule']
        
        state_tuple = tuple(sorted([t.name for t in state['remaining']]))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        for task in state['remaining']:
            if state['time'] + task.duration <= task.deadline:
                new_schedule = state['schedule'] + [task]
                new_remaining = [t for t in state['remaining'] if t != task]
                new_time = state['time'] + task.duration
                new_state = {'schedule': new_schedule, 'remaining': new_remaining, 'time': new_time}
                new_cost = new_time + heuristic(new_state)
                heapq.heappush(queue, (new_cost, new_state))

    return tasks  # Fallback: return unscheduled tasks

# Hill Climbing for priority adjustment
def hill_climbing_priorities(tasks, feedback_sentiment, feedback_topic):
    priorities = {task.topic: 1.0 for task in tasks}
    
    def evaluate(priorities):
        score = sum(task.difficulty * priorities[task.topic] for task in tasks)
        if feedback_sentiment == "negative":
            score += priorities[feedback_topic] * 10
        return score

    for _ in range(100):
        new_priorities = priorities.copy()
        topic = feedback_topic if feedback_sentiment == "negative" else random.choice(list(priorities))
        new_priorities[topic] += 0.1
        if evaluate(new_priorities) > evaluate(priorities):
            priorities = new_priorities
    
    return priorities

# Min-Max for study/leisure balance
def min_max_balance(tasks, max_hours=6):
    def evaluate(state):
        study_hours = sum(t.duration for t in state['study'])
        leisure_hours = state['leisure']
        return study_hours * 2 + leisure_hours

    def min_max(state, depth, is_study):
        if depth == 0 or not state['remaining']:
            return evaluate(state), state

        if is_study:
            best_score, best_state = float('-inf'), None
            for task in state['remaining']:
                new_study = state['study'] + [task]
                new_remaining = [t for t in state['remaining'] if t != task]
                score, _ = min_max({'study': new_study, 'leisure': state['leisure'], 'remaining': new_remaining}, depth-1, False)
                if score > best_score:
                    best_score, best_state = score, {'study': new_study, 'leisure': state['leisure'], 'remaining': new_remaining}
            return best_score, best_state
        else:
            best_score, best_state = float('inf'), None
            for leisure in [0, 1, 2]:
                score, _ = min_max({'study': state['study'], 'leisure': leisure, 'remaining': state['remaining']}, depth-1, True)
                if score < best_score:
                    best_score, best_state = score, {'study': state['study'], 'leisure': leisure, 'remaining': state['remaining']}
            return best_score, best_state

    start = {'study': [], 'leisure': 0, 'remaining': tasks}
    _, best_state = min_max(start, depth=2, is_study=True)
    return best_state

# Parse goals (simplified)
def parse_goals(goals):
    tasks = [
        Task("QA Practice", "QA", 2, 7, 5),
        Task("DILR Practice", "DILR", 2.5, 7, 7),
        Task("VARC Practice", "VARC", 1.5, 7, 4)
    ]
    return tasks

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/goals', methods=['POST'])
def submit_goals():
    goals = request.form['goals']
    tasks = parse_goals(goals)
    proficiency = {"QA": 0.8, "DILR": 0.5, "VARC": 0.7}  # Example
    schedule = {"study": a_star_schedule(tasks, proficiency), "leisure": 1}
    return render_template('schedule.html', schedule=schedule)

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    sentiment, topic = process_feedback(feedback)
    tasks = parse_goals("")  # Reload tasks
    priorities = hill_climbing_priorities(tasks, sentiment, topic)
    # Adjust task durations based on priorities
    for task in tasks:
        task.duration *= priorities[task.topic]
    schedule = min_max_balance(tasks)
    return render_template('feedback.html', sentiment=sentiment, topic=topic, schedule=schedule)

if __name__ == '__main__':
    app.run(debug=True) 
