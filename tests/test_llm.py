import pytest
from src.llm import DeepSeekLLM

@pytest.mark.asyncio
async def test_real_deepseek_request():
    llm = DeepSeekLLM()
    result = await llm.parse_calendar_event("Встреча с клиентом завтра в 15:00 в офисе")
    
    # Проверяем, что получили ответ
    assert result is not None
    
    # Проверяем структуру ответа
    assert "title" in result
    assert "start_time" in result
    assert "end_time" in result
    assert "description" in result
    assert "location" in result
    
    # Проверяем, что даты в ISO формате, если они не null
    assert "T" in result["start_time"]  # start_time обязательно должно быть
    if result["end_time"]:  # end_time может быть null
        assert "T" in result["end_time"]
    
    # Выводим результат для проверки
    print("\nParsed event:")
    print(f"Title: {result['title']}")
    print(f"Start: {result['start_time']}")
    print(f"End: {result['end_time']}")
    print(f"Description: {result['description']}")
    print(f"Location: {result['location']}") 