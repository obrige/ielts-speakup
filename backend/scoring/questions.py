"""
简易题库;MVP 阶段先用静态数据。
实际项目应替换为对接 BC 题库服务的客户端。
"""
QUESTION_BANK: dict[int, list[dict]] = {
    1: [
        {
            "id": "p1-hometown-1",
            "part": 1,
            "topic": "Hometown",
            "prompt": "Let's talk about your hometown. Where are you from?",
            "cues": [
                "What do you like most about your hometown?",
                "Has your hometown changed much since you were a child?",
            ],
            "expected_duration_sec": 60,
        },
        {
            "id": "p1-hobby-1",
            "part": 1,
            "topic": "Hobbies",
            "prompt": "Do you have any hobbies?",
            "cues": [
                "How often do you do this hobby?",
                "Why do you enjoy it?",
            ],
            "expected_duration_sec": 60,
        },
        {
            "id": "p1-reading-1",
            "part": 1,
            "topic": "Reading",
            "prompt": "Do you enjoy reading?",
            "cues": [
                "What kind of books do you prefer?",
                "How much time do you spend reading each week?",
            ],
            "expected_duration_sec": 60,
        },
    ],
    2: [
        {
            "id": "p2-book-1",
            "part": 2,
            "topic": "A Book",
            "prompt": "Describe a book that has had a major influence on you.",
            "cues": [
                "What the book is about",
                "When you read it",
                "Why it is influential",
                "Explain why this book is important to you.",
            ],
            "expected_duration_sec": 120,
        },
    ],
    3: [
        {
            "id": "p3-education-1",
            "part": 3,
            "topic": "Education",
            "prompt": "Let's discuss education in your country.",
            "cues": [
                "How has education changed in your country over the past few decades?",
                "Do you think students today learn differently from previous generations?",
            ],
            "expected_duration_sec": 180,
        },
    ],
}


def get_question(qid: str) -> dict:
    for part in QUESTION_BANK.values():
        for q in part:
            if q["id"] == qid:
                return q
    raise KeyError(f"Question {qid} not found")