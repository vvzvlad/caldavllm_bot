import pytest
from src.llm import DeepSeekLLM

@pytest.mark.asyncio
async def test_real_deepseek_request():
    llm = DeepSeekLLM()
    result = await llm.parse_calendar_event("Meeting with client tomorrow at 15:00 in the office")
    
    # Check that we got a response
    assert result is not None
    
    # Check response structure
    assert "title" in result
    assert "start_time" in result
    assert "end_time" in result
    assert "description" in result
    assert "location" in result
    
    # Check that dates are in ISO format if they are not null
    assert "T" in result["start_time"]  # start_time must be present
    if result["end_time"]:  # end_time can be null
        assert "T" in result["end_time"]
    
    # Print result for verification
    print("\nParsed event:")
    print(f"Title: {result['title']}")
    print(f"Start: {result['start_time']}")
    print(f"End: {result['end_time']}")
    print(f"Description: {result['description']}")
    print(f"Location: {result['location']}") 