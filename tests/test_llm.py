import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from src.llm import DeepSeekLLM

@pytest.fixture
def llm():
    return DeepSeekLLM()

@pytest.mark.asyncio
async def test_parse_calendar_event_today(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом сегодня в 15:00 в офисе")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2024-03-15T15:00:00",
            "end_time": "2024-03-15T16:00:00",
            "description": "Встреча с клиентом",
            "location": "Офис",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_tomorrow(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом завтра в 15:00 в офисе")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2024-03-16T15:00:00",
            "end_time": "2024-03-16T16:00:00",
            "description": "Встреча с клиентом",
            "location": "Офис",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_specific_day(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча в офисе с клиентом 15 числа в 15:00")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2024-03-15T15:00:00",
            "end_time": "2024-03-15T16:00:00",
            "description": "Встреча в офисе с клиентом",
            "location": "Офис",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_specific_date(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом 15 марта в 15:00 в офисе")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2024-03-15T15:00:00",
            "end_time": "2024-03-15T16:00:00",
            "description": "Встреча с клиентом",
            "location": "Офис",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_only_time(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом в 15:00")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2024-03-15T15:00:00",
            "end_time": "2024-03-15T16:00:00",
            "description": "Встреча с клиентом",
            "location": None,
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_past_day(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 20, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом 15-го в 15:00 в офисе")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2024-04-15T15:00:00",
            "end_time": "2024-04-15T16:00:00",
            "description": "Встреча с клиентом",
            "location": "Офис",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_past_date(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 20, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом 15 марта в 15:00 в офисе")
        
        assert result is not None
        assert result == {
            "title": "Встреча с клиентом",
            "start_time": "2025-03-15T15:00:00",
            "end_time": "2025-03-15T16:00:00",
            "description": "Встреча с клиентом",
            "location": "Офис",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_insufficient_info(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом в сентябре")
        
        assert result is not None
        assert result["result"] is False
        assert result["comment"] is not None
        assert "Недостаточно информации" in result["comment"]

@pytest.mark.asyncio
async def test_parse_calendar_event_insufficient_info_with_location(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("Встреча с клиентом в сентябре в офисе")
        
        assert result is not None
        assert result["result"] is False
        assert result["comment"] is not None
        assert "Недостаточно информации" in result["comment"]

@pytest.mark.asyncio
async def test_parse_calendar_event_doctor_appointment(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Добрый день, это ДОКБОТ 🤖
Вы записались на прием в одну из наших клиник DocDeti, DocMed, DocDent по адресу: 121471, Москва г, муниципальный округ Можайский, Петра Алексеева ул, дом 14, помещение 23Н на 24 Марта 2025 г. в 14:20
Врач: Дерматолог Чикатуева Д

Как нас найти: https://clck.ru/34v7dh
Как добраться до клиники:

 •    На общественном транспорте https://vk.com/video-165966750_456239875?list=ln-EjMdQJo7jg345uu27Q
 •    Если вы планируете приехать на машине, сообщите нам её марку, номер и регион для оформления разрешения на временный въезд в ЖК. Без него на личном автомобиле вас не пропустят. Оформить пропуск можно по телефону 
+7 (495) 150 99 51 или в What's App https://wa.me/79855055776""")
        
        assert result is not None
        assert result == {
            "title": "Дерматолог",
            "start_time": "2025-03-24T14:20:00",
            "end_time": "2025-03-24T15:20:00",
            "description": "Прием у дерматолога Чикатуевой Д.",
            "location": "DocDeti, DocMed, DocDent, 121471, Москва г, муниципальный округ Можайский, Петра Алексеева ул, дом 14, помещение 23Н",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_beauty_salon(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
        result = await llm.parse_calendar_event("""Здравствуйте, Влад! 
Вы записаны в «The Kudri club». 

Дата 
06.01.2024 20:00:00 	
Адрес 
Салон красоты «The Kudri club»
город Москва, Духовской пер., дом 17 	
Мастер 
Мастер: Валерия Ан Парикмахерские услуги / Стрижки / Мужская стрижка""")
        
        assert result is not None
        assert result == {
            "title": "Парикмахер",
            "start_time": "2024-01-06T20:00:00",
            "end_time": "2024-01-06T21:00:00",
            "description": "Парикмахерские услуги / Стрижки / Мужская стрижка",
            "location": "Салон красоты «The Kudri club», город Москва, Духовской пер., дом 17",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_online_psychologist(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""💁🏻 Психолог записал вас на сессию 
Не забудьте оплатить вашу сессию до 2025-03-16 19:00, иначе она отменится. 
Перейти к оплате 
Детали сессии: 
• 
Специалист: Татьяна Катаева 
• 
Вид терапии: Индивидуально (онлайн) 
Дата и время сессии: 2025-03-17 19:00 (Московское время) 
• 
Сессия пройдет по видеосвязи: Подробнее в вашем личном кабинете""")
        
        assert result is not None
        assert result == {
            "title": "Психолог",
            "start_time": "2025-03-17T19:00:00",
            "end_time": "2025-03-17T20:00:00",
            "description": "Индивидуально (онлайн)",
            "location": "Онлайн",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_zloydoctor(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Zloydoctor
Москва, Сретенский бульвар д.2
Владислав, здравствуйте! Вы записаны: 18 марта 2024 в 21:15 
Детали визита
Туполев Андрей
zloydocto
Приём zloydoctor 45м new
8 000 ₽""")
        
        assert result is not None
        assert result == {
            "title": "Прием у Туполева Андрея",
            "start_time": "2024-03-18T21:15:00",
            "end_time": "2024-03-18T22:00:00",
            "description": "Приём zloydoctor 45м new",
            "location": "Москва, Сретенский бульвар д.2",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_online_masterclass(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Привет, это Аня из Flow!  😀
Как и договорились, записала на онлайн мастер-класс "Маркетинговая стратегия за 45" или "Как привлекать по 100+ клиентов в В2В бизнес и бизнес в сфере услуг ежемесячно?"
Он пройдет в ближайший понедельник, 5го апреля, в 17:00 и продлится всего 45 минут.""")
        
        assert result is not None
        assert result == {
            "title": "Маркетинговая стратегия за 45",
            "start_time": "2024-04-05T17:00:00",
            "end_time": "2024-04-05T17:45:00",
            "description": "Маркетинговая стратегия за 45 или Как привлекать по 100+ клиентов в В2В бизнес и бизнес в сфере услуг ежемесячно?",
            "location": "Онлайн",
            "result": True,
            "comment": None
        } 