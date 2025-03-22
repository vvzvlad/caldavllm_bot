import pytest
from src.llm import DeepSeekLLM

@pytest.mark.asyncio
async def test_real_deepseek_request():
    llm = DeepSeekLLM()
    
    # Test cases with different date formats
    test_cases = [
        "Meeting with client tomorrow at 15:00 in the office",  # No date specified
        "Meeting with client on 15th at 15:00 in the office",   # Only day specified
        "Meeting with client in September at 15:00 in the office",  # Month specified
        "Meeting with client on March 15th at 15:00 in the office",  # Month and day
    ]
    
    for text in test_cases:
        print(f"\nTesting text: {text}")
        result = await llm.parse_calendar_event(text)
        
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