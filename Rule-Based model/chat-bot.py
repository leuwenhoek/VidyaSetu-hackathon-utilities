from difflib import get_close_matches
import json
import os

def get_knowledge_base(location):
    # Ensure the file exists and contains valid JSON with a 'questions' list
    if not os.path.exists(location) or os.path.getsize(location) == 0:
        with open(location, 'w') as f:
            json.dump({"questions": []}, f, indent=4)

    with open(location, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"questions": []}

    if not isinstance(data, dict):
        data = {"questions": []}

    if "questions" not in data or not isinstance(data.get("questions"), list):
        data["questions"] = []
        with open(location, 'w') as f:
            json.dump(data, f, indent=4)

    return data

def save_knowledge_base(location :str,data :dict):
    with open(location,"w") as f:
        json.dump(data,f,indent=4)

def find_best_match(user_question, questions):
    matches = get_close_matches(user_question,questions,n=1,cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for_question(question,knowledge_base):
    for q in knowledge_base.get('questions', []):
        if q.get('question') == question:
            return q.get('answer')
    return None
        
def chat_bot():
    base_dir = os.path.dirname(__file__)
    location = os.path.join(base_dir, "knowledge_base.json")
    knowledge_base = get_knowledge_base(location)

    while True:
        user_input = input("You : ")

        if user_input.lower() == "quit":
            break

        question_texts = [q.get('question') for q in knowledge_base.get('questions', []) if isinstance(q, dict) and 'question' in q]

        best_match = find_best_match(user_input, question_texts)

        if best_match:
            answer = get_answer_for_question(best_match, knowledge_base)
            if answer:
                print(f"Bot: {answer}")
            else:
                print("Bot: I don't have an answer stored for that question.")
        else:
            print("Bot: I don't know the answer. Can you teach me?")
            new_ans = input("Type the answer or 'skip' to skip: ")

            if new_ans.lower() != "skip":
                knowledge_base['questions'].append({"question": user_input, "answer": new_ans})
                save_knowledge_base(location, knowledge_base)
                print("Bot: Thank you! I learned a new response.")

def main():
    location = os.path.join(os.getcwd(),"Rule-Based model","knowledge_base.json")
    get_knowledge_base(location) 
    return 0

if __name__ == "__main__":
    chat_bot()