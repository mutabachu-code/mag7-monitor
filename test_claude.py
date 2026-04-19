import anthropic

client = anthropic.Anthropic(api_key="sk-ant-your-key-here")

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[{
        "role": "user",
        "content": "Analyse this: AAPL at $270, RSI 28, MACD bullish, above SMA200, volume surge 1.3x. BUY or SELL or HOLD? Reply in one sentence."
    }]
)

print(response.content[0].text)
