import requests
import json

url = "http://127.0.0.1:5000/evaluate"
payload = {
    "prompt": "Some people think children should do housework. Discuss both views and give your opinion.",
    "essay": "Many people argue that children should do housework because it teaches them responsibility and practical skills for the future. On the other hand, some believe that children should focus on their studies and enjoy their childhood without the burden of household chores. In my opinion, a balanced approach is best. Children can contribute to simple tasks, which helps them learn important life skills, while still having enough time for education and play. This ensures they develop into well-rounded individuals.",
    "debug": True
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error testing API: {e}")
