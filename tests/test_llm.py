import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from src.llm import DeepSeekLLM
import os

def check_api_key():
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.fail("API_KEY environment variable is not set. Please set it before running tests.")

@pytest.fixture(autouse=True)
def setup():
    check_api_key()

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
            "description": "Сессия с психологом Татьяной Катаевой",
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
            "title": "Доктор Туполев",
            "start_time": "2024-03-18T21:15:00",
            "end_time": "2024-03-18T22:00:00",
            "description": "Прием у доктора Туполева",
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
Он пройдет в ближайший понедельник, 5го апреля, в 17:00 и продлится всего 45 минут.
🔥 Ссылку на мастер-класс мы скинем в день мероприятия в нашем Телеграм-канале https://cutt.ly/cciPpjb - подписывайтесь на него прямо сейчас!""")
        
        assert result is not None
        assert result == {
            "title": "Маркетинговая стратегия за 45",
            "start_time": "2024-04-05T17:00:00",
            "end_time": "2024-04-05T17:45:00",
            "description": "Маркетинговая стратегия за 45 или Как привлекать по 100+ клиентов в В2В бизнес и бизнес в сфере услуг ежемесячно?",
            "location": "https://cutt.ly/cciPpjb",
            "result": True,
            "comment": None
        }

@pytest.mark.asyncio
async def test_parse_calendar_event_ceramic_breakfast(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Санкт-Петербург 
Добрый вечер! 
Приглашаю на КЕРАМИЧЕСКИЙ ЗАВТРАК☺️
На встрече В ЭТУ ПЯТНИЦУ поговорим об участии в маркетах/фестивалях/выставках. Будет полезно не только керамистам, но и всем хендмейд-мастерам. 
Приглашённый гость - Аня Горфинкель, организатор фестиваля «Ceramania»❤️
Аня расскажет о подготовке к маркету, как оформить стенд, как выделится среди участников и какая перспектива участия? Поделится огромнейшим опытом, расскажет как проходит Ceramania и ответит на ваши вопросы.
Очень полезная встреча, обсудим инструменты, которые помогут в развитии вас и вашего бренда.
Будут угощения, чай, кофе, полезные знакомства, горячие обсуждения и обмен опытом. В общем, настоящий нетворкинг среди керамистов. 
Ждём вас в гости в нашей чудесной студии с видом на залив🌊
Когда: 17 марта (пт)
Время: 12:00
Стоимость: 800₽ 
Где: СПб, Севкабель порт, Кожевенная линия 34а
Пишите в личку для записи❤️""")
                
        assert result is not None
        assert result["title"] == "Керамический завтрак"
        assert result["start_time"] == "2024-03-17T12:00:00"
        assert result["end_time"] == "2024-03-17T13:00:00"
        assert result["location"] == "СПб, Севкабель порт, Кожевенная линия 34а"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_webinar(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 2, 20, 12, 0)
        result = await llm.parse_calendar_event("""Добрый день.
Приглашаю на бесплатный вебинар по теме «Разработка ПО для GigaDevice GD32 семейства. Профессиональный подход.
Разработка на базе открытого ПО.»
Ссылка на вебинар: https://go.mywebinar.com/smkx-fnrj-qpbm-kfdb
Содержание вебинара: https://dab-embedded.com/en/services/webinar-gigadevice-gd32-software-dev-prof/?lang=en
Специально для тренинга была разработана плата на базе GigaDevice GD32F470 и FT2232H (в качестве отладчика) - фото.
25.Февраля 2023, 15:00 по Москве
Длительность 1 час.
#вебинар #firmware #GD32 #opensource""")
        
        assert result is not None
        assert result["title"] == "Разработка ПО для GigaDevice GD32 семейства"
        assert result["start_time"] == "2023-02-25T15:00:00"
        assert result["end_time"] == "2023-02-25T16:00:00"
        assert result["description"] == "Разработка ПО для GigaDevice GD32 семейства. Профессиональный подход. Разработка на базе открытого ПО."
        assert result["location"] == "https://go.mywebinar.com/smkx-fnrj-qpbm-kfdb"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_conference(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2022, 12, 1, 12, 0)
        result = await llm.parse_calendar_event("""CCCP-2022
Институт когнитивных нейронаук ВШЭ приглашает принять участие в ежегодной конференции
Cortex and Cognition: Connection Principles. Neuroimaging and clinical applications (CCCP-2022)
21-22 декабря 2022 года
Темы конференции: 
- инвазивные и неинвазивные методы нейрокартирования (MEG, EEG, fMRI)
- стимуляция (TMS, tDCS, tACS)
- оптогенетика
- термогенетика
- клиническое применения методов нейровизуализации 
- и др. 
Конференция пройдет в смешанном формате (очно + онлайн).
Язык конференции -- английский.
Заявки на участие (с тезисами) принимаются по этой ссылке до 11.12.2022 включительно.""")
        
        assert result is not None
        assert result["title"] == "Cortex and Cognition: Connection Principles"
        assert result["start_time"] == "2022-12-21T00:00:00"
        assert result["end_time"] == "2022-12-22T23:59:59"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_skoltech_conference(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2021, 11, 1, 12, 0)
        result = await llm.parse_calendar_event("""📢📢📢Skoltech Neuro и CNBR_Open приглашают к участию в мини-конференции "Neuroenhancement and Neuromodulation"  
Ведущие специалисты из нескольких исследовательских организаций: Skoltech, Mayo Clinic (USA), КФУ (Казань), Университета Лобачевского (Нижний Новгород), Иннополиса (Казань) и др. встретятся для того чтобы обсудить свежие темы в области нейромодуляции и возможности совместных проектов.  
🗓 24 Ноября 2021, 10:00 —16:00  
📍 Skoltech Campus, E-R2-2019  
💰 Свободный доступ, политика COVID-FREE (требуются сертификат о вакцинации, антителах или свежий  PCR-test )  
🗣 Язык – English, Русский 
Регистрация (обязательна): по ссылке. 
Список выступлений с абстрактами можно посмотреть здесь""")
        
        assert result is not None
        assert result["title"] == "Neuroenhancement and Neuromodulation"
        assert result["start_time"] == "2021-11-24T10:00:00"
        assert result["end_time"] == "2021-11-24T16:00:00"
        assert result["result"] is True
        assert result["comment"] is None 

@pytest.mark.asyncio
async def test_parse_calendar_event_meg_webinar(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Некоторое время я тут советовался насчет того, кого бы пригласить на новый регулярный вебинар нейрокогнитивного направления. В результате столкновений с организационно-бюрократической реальностью исходную идею пришлось несколько приземлить, тем не менее, несколько вебинаров МЭГ-центра МГППУ в этом году непременно пройдут. Мы уже договорились с предложенным здесь Юрием Павловым (ManyEEGLabs), он выступит в мае. 
А первый вебинар будет 7 апреля, 18:00 по Москве. На нем профессор Гарвардского и Гетеборгского университетов Ношин Хаджикхани (Nouchine Hadjikhani) расскажет про Eye contact in autism and its link to the imbalance of excitation and inhibition in the brain.""")
        
        assert result is not None
        assert result["start_time"] == "2024-04-07T18:00:00"
        assert result["end_time"] == "2024-04-07T19:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_audio_conference(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 4, 10, 12, 0)
        result = await llm.parse_calendar_event("""Сегодня, в субботу, 10 апреля, в 22:22 по иркутскому времени (в столице будет на пять часов меньше) на этом канале будет аудиоконференция. «Клубхаус», как говорит молодёжь.
Приглашаю желающих послушать неофициальную презентацию Манифеста бумажной книги, который готовит наше творческое объединение. Мы планируем его опубликовать этой весной, но сперва хочется потестировать его на людях.""")
        
        assert result is not None
        assert result["start_time"] == "2024-04-10T17:22:00"
        assert result["end_time"] == "2024-04-10T18:22:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_birthday(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Мои дорогие любимые люди! 
Я приглашаю вас всех повторно на мой юбилейчик в субботу вечером) 
Забронировала столик тут в 18:00 в эту субботу: https://yandex.com/maps/org/8_oz/1171896955
Всех вас жду ❤""")
        
        assert result is not None
        assert result["start_time"] == "2024-03-16T18:00:00"
        assert result["end_time"] == "2024-03-16T19:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_travel(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""⚡ Молния!
Россия 17 мая открыла сухопутную границу с Грузией. А это значит...
Rock'n'road едет в Грузию!
Друзья, мы ждали этого 2 года! Приглашаем вас в автопутешествие в страну гор и вина ⛰🍷
Мы запланировали поездку:
20–28 августа (9 дней, с субботы по воскресенье следующей недели) со стартом из Мин Вод. 
Посетим самые яркие и знаковые места Грузии: Степанцминда, Тбилиси, Мцхета, Боржоми, Местиа.
🤟 На первую поездку-разведку 20–28 августа действует специальная цена для своих: 59 900.
Сентябрьская поездка — 65 900 💰
Завтра мы анонсируем эти поездки в канале, так что торопитесь 😉
Писать, бронировать и задавать вопросы можно в личку  @rock_n_road
Для путешествия понадобится сертификат о вакцинации любой двухкомпонентной вакциной или результат ПЦР-теста, сделанного в течение 72 часов до въезда.""")
        
        assert result is not None
        assert result["start_time"] == "2024-08-20T00:00:00"
        assert result["end_time"] == "2024-08-28T23:59:59"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_poly_date_club(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""Хочу объявить ещё об одном мероприятии — надеюсь, что формат зайдёт и будет регулярным, но кто знает — поэтому лучше ловите момент и регистрируйтесь сейчас! Это первое большое мероприятие от меня и у меня дома, я очень волнуюсь и очень-очень жду всех подписчиков ☀️
26 мая (в этот четверг) состоится Poly Date Club — безопасное пространство с атмосферой добра и принятия! 
Приглашаем всех, кто хочет почувствовать себя частью секс-позитивного комьюнити, найти новые знакомства по теме, и просто хорошо провести время среди like-minded people. 
Мы за открытое общение, знакомства на вербальном и физическом уровне, принятие. 
Важно: это не кинк-пати и заниматься сексом на этой встрече нельзя. 
Можно лайт взаимодействия - объятия, тактильности. 
Мы создаём безопасное пространство для знакомств и самовыражения для полиаморов, тематиков и всех желающих стать частью местного секс-позитивного комьюнити. 
Наши принципы: 
LGBTQ+ friendly
Poly friendly 
«Да - значит да, нет - значит нет»""")
        
        assert result is not None
        assert result["title"] == "Poly Date Club"
        assert result["start_time"].startswith("2024-05-26T")
        assert result["end_time"].startswith("2024-05-26T")
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_night_party(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm.parse_calendar_event("""⚠️Warning ⚠️ 
Анонсирую тусу у меня в квартире в Спб 
Поэтому приглашаю всех к себе 27-го августа в 20:00 до 05:00 28-го августа на тусовочу.
Будет много вкусной еды, алкоголь(сидры, коктейли если хотите крепче — несите сами). 
Поболтаем, расскажем куллстори друг другу, будет весело 🎊
Если хотите взять +1, напишите мне об этом пожалуйста и туда же за адресом, кто не знает куда ехать
С нетерпением буду ждать каждого""")
        
        assert result is not None
        assert result["start_time"] == "2024-08-27T20:00:00"
        assert result["end_time"] == "2024-08-28T05:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_outdoor_concert(llm):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 7, 15, 12, 0)
        result = await llm.parse_calendar_event("""📅 Когда: 31 июля 2023, 20:00
Добро пожаловать в мир музыки и волшебства!
Мы с радостью приглашаем вас на незабываемый концерт, который пройдет на удивительной площадке под открытым небом . Это будет вечер, наполненный музыкой и вдохновением. Представьте себе, как звуки музыки сочетаются с прикосновением летнего бриза, создавая магию эмоций и перенося нас в другой мир.
Наш большой музыкальный состав сыграет для вас программу Miyazaki Dreams, и каждая нота проникнет в вашу душу.
🚩 Адрес: Ереван, ул. Арами, 42
💰Цена: 5000 ֏""")
        
        assert result is not None
        assert result["title"] == "Концерт"
        assert result["start_time"] == "2023-07-31T20:00:00"
        assert result["end_time"] == "2023-07-31T21:00:00"
        assert "Miyazaki Dreams" in result["description"]
        assert result["location"] == "Ереван, ул. Арами, 42"
        assert result["result"] is True
        assert result["comment"] is None