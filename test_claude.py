import anthropic
import os

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say: Claude is connected to Mag7 dashboard"}]
)

print(response.content[0].text)
