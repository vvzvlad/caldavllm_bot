# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name, protected-access
# pylance: disable=reportMissingImports, reportMissingModuleSource, reportGeneralTypeIssues
# type: ignore

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from src.llm import DeepSeekLLM
import os
from unittest.mock import MagicMock

def check_api_key():
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.fail("API_KEY environment variable is not set. Please set it before running tests.")

@pytest.fixture(autouse=True)
def setup():
    check_api_key()

@pytest.fixture
def llm_instance():
    return DeepSeekLLM()


@pytest.mark.asyncio
async def test_datetime_mock():
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 20, 12, 0)
        llm = DeepSeekLLM()
        current_date = llm._return_datetime()
        assert current_date == datetime(2024, 3, 20, 12, 0)
        assert current_date != datetime(2024, 3, 20, 12, 1)


@pytest.mark.asyncio
async def test_generate_calendar():
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 20, 12, 0)
        
        llm = DeepSeekLLM()
        calendar = await llm._generate_calendar()
        expected_calendar = """20 March — среда (сегодня)
21 March — этот четверг
22 March — эта пятница
23 March — эта суббота
24 March — это воскресенье
25 March — следующий понедельник
26 March — следующий вторник
27 March — следующая среда
28 March — следующий четверг
29 March — следующая пятница
30 March — следующая суббота
31 March — следующее воскресенье
01 April — следующий понедельник
02 April — следующий вторник"""
        assert calendar == expected_calendar

@pytest.mark.asyncio
async def test_parse_calendar_event_today(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем сегодня в 15:00 в офисе")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2024-03-15T15:00:00"
        assert result["end_time"] == "2024-03-15T16:00:00"
        assert result["location"] == "Офис"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_tomorrow(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем завтра в 15:00 в офисе")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2024-03-16T15:00:00"
        assert result["end_time"] == "2024-03-16T16:00:00"
        assert result["location"] == "Офис"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_specific_day(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча в офисе с сергеем 15 числа в 15:00")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2024-03-15T15:00:00"
        assert result["end_time"] == "2024-03-15T16:00:00"
        assert result["location"] == "Офис"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_specific_date(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем 15 марта в 15:00 в офисе")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2024-03-15T15:00:00"
        assert result["end_time"] == "2024-03-15T16:00:00"
        assert result["location"] == "Офис"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_only_time(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем в 15:00")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2024-03-15T15:00:00"
        assert result["end_time"] == "2024-03-15T16:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_past_day(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 20, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем 15-го в 15:00 в офисе")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2024-04-15T15:00:00"
        assert result["end_time"] == "2024-04-15T16:00:00"
        assert result["location"] == "Офис"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_past_date(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 20, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем 15 марта в 15:00 в офисе")
        
        assert result is not None
        assert "серге" in result["title"].lower()
        assert result["start_time"] == "2025-03-15T15:00:00"
        assert result["end_time"] == "2025-03-15T16:00:00"
        assert result["location"] == "Офис"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_insufficient_info(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем в сентябре")
        
        assert result is not None
        assert result["result"] is False
        assert result["comment"] is not None
        assert "Недостаточно информации" in result["comment"]

@pytest.mark.asyncio
async def test_parse_calendar_event_insufficient_info_with_location(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("Встреча с сергеем в сентябре в офисе")
        
        assert result is not None
        assert result["result"] is False
        assert result["comment"] is not None
        assert "Недостаточно информации" in result["comment"]

@pytest.mark.asyncio
async def test_parse_calendar_event_doctor_appointment(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Добрый день, это ДОКБОТ 🤖
Вы записались на прием в одну из наших клиник DocDeti, DocMed, DocDent по адресу: 121471, Москва г, муниципальный округ Можайский, Петра Алексеева ул, дом 14, помещение 23Н на 24 Марта 2025 г. в 14:20
Врач: Дерматолог Чикатуева Д

Как нас найти: https://clck.ru/34v7dh
Как добраться до клиники:

•    На общественном транспорте https://vk.com/video-165966750_456239875?list=ln-EjMdQJo7jg345uu27Q
•    Если вы планируете приехать на машине, сообщите нам её марку, номер и регион для оформления разрешения на временный въезд в ЖК. Без него на личном автомобиле вас не пропустят. Оформить пропуск можно по телефону 
+7 (495) 150 99 51 или в What's App https://wa.me/79855055776""")
        
        assert result is not None
        assert "дерматолог" in result["title"].lower()
        assert result["start_time"] == "2025-03-24T14:20:00"
        assert result["end_time"] == "2025-03-24T15:20:00"
        assert "121471, Москва г" in result["location"]
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_beauty_salon(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""Здравствуйте, Влад! 
Вы записаны в «The Kudri club». 

Дата 
06.01.2024 20:00:00 	
Адрес 
Салон красоты «The Kudri club»
город Москва, Духовской пер., дом 17 	
Мастер 
Мастер: Валерия Ан Парикмахерские услуги / Стрижки / Мужская стрижка""")
        
        assert result is not None
        assert "стрижка" in result["title"].lower()
        assert result["start_time"] == "2024-01-06T20:00:00"
        assert result["end_time"] == "2024-01-06T21:00:00"
        assert "Духовской" in result["location"]
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_online_psychologist(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""💁🏻 Психолог записал вас на сессию 
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
        assert "сессия" in result["title"].lower() or "психолог" in result["title"].lower()
        assert result["start_time"] == "2025-03-17T19:00:00"
        assert result["end_time"] == "2025-03-17T20:00:00"
        assert result["location"] == "Онлайн"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_zloydoctor(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Zloydoctor
Москва, Сретенский бульвар д.2
Владислав, здравствуйте! Вы записаны: 18 марта 2024 в 21:15 
Детали визита
Туполев Андрей
zloydocto
Приём zloydoctor 45м new
8 000 ₽""")
        
        assert result is not None
        assert "Туполев" in result["title"]
        assert result["start_time"] == "2024-03-18T21:15:00"
        assert result["end_time"] == "2024-03-18T22:00:00"
        assert "Сретенский бульвар" in result["location"]
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_online_masterclass(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Привет, это Аня из Flow!  😀
Как и договорились, записала на онлайн мастер-класс "Маркетинговая стратегия за 45" или "Как привлекать по 100+ клиентов в В2В бизнес и бизнес в сфере услуг ежемесячно?"
Он пройдет в ближайший понедельник, 5го апреля, в 17:00 и продлится всего 45 минут.
🔥 Ссылку на мастер-класс мы скинем в день мероприятия в нашем Телеграм-канале https://cutt.ly/cciPpjb - подписывайтесь на него прямо сейчас!""")
        
        assert result is not None
        assert "маркетинговая стратегия" in result["title"].lower()
        assert result["start_time"] == "2024-04-05T17:00:00"
        assert result["end_time"] == "2024-04-05T17:45:00"
        assert result["location"] == "https://cutt.ly/cciPpjb"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_ceramic_breakfast(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Санкт-Петербург 
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
        assert "керамический завтрак" in result["title"].lower()
        assert result["start_time"] == "2024-03-17T12:00:00"
        assert result["end_time"] == "2024-03-17T13:00:00"
        assert "Севкабель" in result["location"]
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_webinar(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 2, 20, 12, 0)
        result = await llm_instance.parse_calendar_event("""Добрый день.
Приглашаю на бесплатный вебинар по теме «Разработка ПО для GigaDevice GD32 семейства. Профессиональный подход.
Разработка на базе открытого ПО.»
Ссылка на вебинар: https://go.mywebinar.com/smkx-fnrj-qpbm-kfdb
Содержание вебинара: https://dab-embedded.com/en/services/webinar-gigadevice-gd32-software-dev-prof/?lang=en
Специально для тренинга была разработана плата на базе GigaDevice GD32F470 и FT2232H (в качестве отладчика) - фото.
25.Февраля 2023, 15:00 по Москве
Длительность 1 час.
#вебинар #firmware #GD32 #opensource""")
        
        assert result is not None
        assert "GD32" in result["title"]
        assert result["start_time"] == "2023-02-25T15:00:00"
        assert result["end_time"] == "2023-02-25T16:00:00"
        assert result["location"] == "https://go.mywebinar.com/smkx-fnrj-qpbm-kfdb"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_conference(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2022, 12, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""CCCP-2022
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
        assert result["start_time"] == "2022-12-21T10:00:00"
        assert result["end_time"] == "2022-12-22T23:59:59"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_skoltech_conference(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2021, 11, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""📢📢📢Skoltech Neuro и CNBR_Open приглашают к участию в мини-конференции "Neuroenhancement and Neuromodulation"  
Ведущие специалисты из нескольких исследовательских организаций: Skoltech, Mayo Clinic (USA), КФУ (Казань), Университета Лобачевского (Нижний Новгород), Иннополиса (Казань) и др. встретятся для того чтобы обсудить свежие темы в области нейромодуляции и возможности совместных проектов.  
🗓 24 Ноября 2021, 10:00 —16:00  
📍 Skoltech Campus, E-R2-2019  
💰 Свободный доступ, политика COVID-FREE (требуются сертификат о вакцинации, антителах или свежий  PCR-test )  
🗣 Язык – English, Русский 
Регистрация (обязательна): по ссылке. 
Список выступлений с абстрактами можно посмотреть здесь""")
        
        assert result is not None
        assert result["start_time"] == "2021-11-24T10:00:00"
        assert result["end_time"] == "2021-11-24T16:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_meg_webinar(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Некоторое время я тут советовался насчет того, кого бы пригласить на новый регулярный вебинар нейрокогнитивного направления. В результате столкновений с организационно-бюрократической реальностью исходную идею пришлось несколько приземлить, тем не менее, несколько вебинаров МЭГ-центра МГППУ в этом году непременно пройдут. Мы уже договорились с предложенным здесь Юрием Павловым (ManyEEGLabs), он выступит в мае. 
А первый вебинар будет 7 апреля, 18:00 по Москве. На нем профессор Гарвардского и Гетеборгского университетов Ношин Хаджикхани (Nouchine Hadjikhani) расскажет про Eye contact in autism and its link to the imbalance of excitation and inhibition in the brain.""")
        
        assert result is not None
        assert result["start_time"] == "2024-04-07T18:00:00"
        assert result["end_time"] == "2024-04-07T19:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_audio_conference(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 4, 10, 12, 0)
        result = await llm_instance.parse_calendar_event("""Сегодня, в субботу, 10 апреля, в 22:22 по иркутскому времени (в столице будет на пять часов меньше) на этом канале будет аудиоконференция. «Клубхаус», как говорит молодёжь.
Приглашаю желающих послушать неофициальную презентацию Манифеста бумажной книги, который готовит наше творческое объединение. Мы планируем его опубликовать этой весной, но сперва хочется потестировать его на людях.""")
        
        assert result is not None
        assert result["start_time"] == "2024-04-10T17:22:00"
        assert result["end_time"] == "2024-04-10T18:22:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_birthday(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 3, 19, 12, 0)
        result = await llm_instance.parse_calendar_event("""
[Екатерина Муринова, 19.03.2025, 13:32]:
Мои дорогие любимые люди! 
Я приглашаю вас всех повторно на мой юбилейчик в субботу вечером) 
Забронировала столик тут в 18:00 в эту субботу: https://yandex.com/maps/org/8_oz/1171896955
Всех вас жду ❤""")
        
        assert result is not None
        assert result["start_time"] == "2025-03-22T18:00:00"
        assert result["end_time"] == "2025-03-22T19:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_travel(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""⚡ Молния!
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
async def test_parse_calendar_event_poly_date_club(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Хочу объявить ещё об одном мероприятии — надеюсь, что формат зайдёт и будет регулярным, но кто знает — поэтому лучше ловите момент и регистрируйтесь сейчас! Это первое большое мероприятие от меня и у меня дома, я очень волнуюсь и очень-очень жду всех подписчиков ☀️
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
        assert "Poly Date Club" in result["title"]
        assert result["start_time"].startswith("2024-05-26T")
        assert result["end_time"].startswith("2024-05-26T")
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_night_party(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""⚠️Warning ⚠️ 
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
async def test_parse_calendar_event_outdoor_concert(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 7, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""📅 Когда: 31 июля 2023, 20:00
Добро пожаловать в мир музыки и волшебства!
Мы с радостью приглашаем вас на незабываемый концерт, который пройдет на удивительной площадке под открытым небом . Это будет вечер, наполненный музыкой и вдохновением. Представьте себе, как звуки музыки сочетаются с прикосновением летнего бриза, создавая магию эмоций и перенося нас в другой мир.
Наш большой музыкальный состав сыграет для вас программу Miyazaki Dreams, и каждая нота проникнет в вашу душу.
🚩 Адрес: Ереван, ул. Арами, 42
💰Цена: 5000 ֏""")
        
        assert result is not None
        assert "Концерт" in result["title"]
        assert result["start_time"] == "2023-07-31T20:00:00"
        assert "Miyazaki Dreams" in result["description"]
        assert result["location"] == "Ереван, ул. Арами, 42"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_pcb_webinar(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 10, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""🎓 Продолжаем цикл наших вебинаров по печатным платам.
📌 Ждём вас 17 октября в 11:00!
Рассмотрим два вопроса:
💬 Спецификация ГРАН и сертификат соответствия.
проведем обзор требований, разработанных нами для обеспечения качества печатных плат;
расскажем о том, как читать и понимать сертификат соответствия — документ, который приходит вместе с каждой поставкой печатных плат.""")
        
        assert result is not None
        assert result["start_time"] == "2024-10-17T11:00:00"
        assert result["end_time"] == "2024-10-17T12:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_japan_day(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 11, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""11.11 11:00–15:00 День Японии в LAN
Приглашаем вас на творческую встречу, где вас ждет лекция о цветовом символизме в Японии и художественный воркшоп, на котором будем рисовать картины с японской палитрой цветов.
Встреча состоит из двух частей: лекция и художественный воркшоп.
На лекции рассмотрим цветовой символизм в Японии, включая:
- цвета, связанные с лисой, бамбуком и васаби
- причина ярких и разноцветных волос у героев аниме
- предпочитаемые оттенки нарядов японских благородных особ и цвет одежды на императоре Японии.
На воркшопе рассмотрим традиционную японскую систему цветов и создадим картину в японском стиле с помощью уникальной цветовой палитры. Уровень сложности: минимальный, подходит даже для начинающих.
Встречу проводят:
Валерия Прокаева — преподаватель японского языка, автор лекций о языке и культуре Японии и Алёна Красильникова — художница, педагог, основательница школы Go art.
Группа небольшая и ламповая, успейте забронировать место.
Цена: 12 000 AMD с человека, 22 000 AMD с двоих (нужна предварительная регистрация)
50% предоплата, после регистрации мы с вами свяжемся для уточнения деталей оплаты
Место: кафе «LAN» (ул. Туманяна, 35Г)""")
        
        assert result is not None
        assert result["start_time"] == "2024-11-11T11:00:00"
        assert result["end_time"] == "2024-11-11T15:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_job_interview(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Olga,
Владислав, добрый день. Прощу прощения. прибоелела и на пару дней вывалилась и процесса.
если подробнее, то у меня есть вот такая информация: Кратко о вакансии. Функционал - отвечать за раззработку всех собственных девайсов Мтс. От умных ошейников, до колонок. Мтс рассматривает возможность создания подразделения по проектированию созданию и дистрибуции умных девайсов. И нам нужен руководитель данного подразделения. 
По мимо разработки дизайна, заказ партии промышленной ее ввоз и дистрибуция в РФ.
На данный момент мы веде такой совсем закрытый поиск, поэтому описания вакансии к сожалению не имею. э
Olga, 
Коллеги из департамента очень хотят пригласить вас на зум )
vvzvlad,
Добрый день! Звучит интересно
Давайте, я с удовольствием пообщаюсь.
Olga,
Супер! У вас какие есть слоты по времени ?
Я сейчас проверю слоты у руководителя департамента в календаре и отпишу вам
 Понедельник, 18 числло, 16;00 часов""")
        
        assert result is not None
        assert result["start_time"] == "2024-03-18T16:00:00"
        assert result["end_time"] == "2024-03-18T17:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_gokon(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 12, 24, 10, 0)
        result = await llm_instance.parse_calendar_event("""Влад, добрый день! Это свидания GOKON! на сегодня сформирована группа девушек возраста 30-34) красивые,интересные, у вас есть общие интересы! 19:00, м.Курская . может быть смогли бы присоединиться?)
Мы готовы пригласить тебя на вечер Gokon. Группа сформирована💙 
Для того чтобы подтвердить свое участие, необходимо сделать взнос в течение 2-х часов в 100% объеме. Стоимость участия 3600 рублей.
Можно оплатить переводом на карту Сбербанк. Получатель Дмитрий Сергеевич Щ. Если у тебя комиссия, запроси ссылку на виртуальную кассу.
После оплаты пришли, пожалуйста, чек.""")
        
        assert result is not None   
        assert result["start_time"] == "2023-12-24T19:00:00"
        assert "Курская" in result["location"]
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_hamovniki(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 11, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""09.11 Суббота в 16:50
Гуляем по Хамовникам! Планируется посещение арт-пространства "Солодовня". 
Глянем:
⭐️ Как появилась фабрика Красная Роза? 
🎻 Почему Лев Толстой купил здесь усадьбу? Какова роль фабрики в его жизни?
👩‍🦽Чем интересен секретарь Толстого Валентин Булгаков? 
🧶 Как работал "комитет по призрению нищих"? Чего комитет добился?""")
        
        assert result is not None
        assert result["start_time"] == "2024-11-09T16:50:00"
        assert result["end_time"] == "2024-11-09T17:50:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_hpmor(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""💥 Празднуем десятилетие «Гарри Поттера и методов рационального мышления» в Москве 💥
📅 Дата: суббота, 16 марта
⏰ Время: 16:00
📍 Место: г. Москва, адрес будет указан в приглашении
💰 Стоимость: участие бесплатное
Почти десять лет назад, 14 марта 2015 года, Юдковский опубликовал последнюю главу «Гарри Поттера и методов рационального мышления». Тогда по всему миру люди собрались, чтобы отпраздновать завершение книги.""")
        
        assert result is not None
        assert result["start_time"] == "2024-03-16T16:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_birthday_invite(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Спешу напомнить, что у меня 17-го декабря день рождения и я хотела бы пригласить тебя в гости на его празднование
Где: 
Санкт-Петербург, Прова 34, кв.2
Когда:
15 декабря в 17:00
Подскажи пожалуйста, ждать ли мне тебя?
Если да, зайди пожалуйста в чатик по др, чтобы я могла понимать сколько у меня будет человеков""")
        
        assert result is not None
        assert result["start_time"] == "2024-12-15T17:00:00"
        assert result["end_time"] == "2024-12-15T18:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_mai_career_fair(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Московский авиационный институт организовывает встречу  предприятий по проектированию и производству электроники со студентами конструкторских направлений. 21 апреля в 17:00 проводится профориентационное мероприятие  с целью знакомства студентов с предприятиями-работодателями в сфере электроники. Можно рассказать о возможностях для студентах, взять на работу или стажировку (платную или бесплатную) или договориться о возможности прохождения практики на базе компании. 
Мероприятие проходит в очном формате по адресу Москва, Волоколамское шоссе, дом 4, корпус 6, Главный учебный корпус МАИ. 
Приглашаем организации выступить перед студентами и рассказать о себе. 
Узнать подробности и подтвердить участие в мероприятии можно до 7 апреля по почте platforma@mai.ru и по телефону +7 (977) 551-77-41 (тг @ckp_mai), Алёна Хабичева""")
        
        assert result is not None
        assert result["start_time"] == "2024-04-21T17:00:00"
        assert result["end_time"] == "2024-04-21T18:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_online_interview(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2022, 4, 6, 12, 0)
        result = await llm_instance.parse_calendar_event("""Влад, добрый день!
Меня зовут Надежда-редактор Вечерней Москвы. 
У нас сегодня состоится прямой эфир буквально на 20 мин, и я бы хотела пригласить Вас, как спикера) Подключение онлайн.
Время: 17.15-17.35
Тема: Новый пакет санкций от ЕС и США 
Вопросы:
- «ввести точечные запреты на экспорт в Россию на сумму 10 млрд евро, в частности в области полупроводников»:  
По поводу полупроводников - надо уже разобраться, есть они у нас или нет? Какие есть ответные ходы и варианты замещения?
Скажите, пожалуйста,это Ваша экспертная тема?""")
        
        assert result is not None
        assert result["start_time"] == "2022-04-06T17:15:00"
        assert result["end_time"] == "2022-04-06T17:35:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_highload_committee(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2022, 2, 1, 12, 0)
        result = await llm_instance.parse_calendar_event("""Привет, друзья!
Мы уже начали работать над летними конференциями.
24 февраля в 11:00 (МСК) состоится встреча с Программным комитетом Saint HighLoad++ 2022.
Для участия нужно зарегистрироваться. Перед началом мероприятия на указанную вами почту придёт ссылка на ZOOM.
Подробности и список участников от Программного комитета — по ссылке. https://onticolist.us8.list-manage.com/track/click?u=719c4e65585ea6013f361815e&id=a4424600c4&e=c06b461e7a""")
        
        assert result is not None
        assert result["start_time"] == "2022-02-24T11:00:00"
        assert result["end_time"] == "2022-02-24T12:00:00"
        assert result["location"] == "https://onticolist.us8.list-manage.com/track/click?u=719c4e65585ea6013f361815e&id=a4424600c4&e=c06b461e7a"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_vk_teams_webinar(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Тренд на суперапп: как изменится рынок коммуникационных сервисов в 2024 году и что нового появилось в VK Teams
Приглашаем вас на вебинар, посвященный тенденциям рынка корпоративных коммуникационных сервисов и обзору нового функционала супераппа VK Teams. 
Подключайтесь к нам 15 ноября в 17:00 . 
Предварительно обязательно зарегистрируйтесь на сайте.
Зарегистрироваться https://mailer.mail.ru/pub/mailer/click/21881/eyJhbGciOiJ""")
        
        assert result is not None
        assert result["start_time"] == "2024-11-15T17:00:00"
        assert result["end_time"] == "2024-11-15T18:00:00"
        assert result["location"] == "https://mailer.mail.ru/pub/mailer/click/21881/eyJhbGciOiJ"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_tax_consultation(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Факультет управления финансами и права приглашает главных бухгалтеров, финансовых директоров и руководителей компаний, собственников бизнеса, налоговых юристов, специалистов по налоговому планированию принять участие в практической консультации или онлайн-трансляции
Трансформация налоговой системы в 2024-2025 годах.
Оценка рисков с целью недопущения мероприятий налогового контроля. Арбитражная практика и алгоритм прохождения налоговой проверки
20 сентября 2024 года
г. Москва, ТГК «Измайлово», Отель «Вега»
Два формата участия: очное присутствие или онлайн-трансляция
Лектор: Ряховский Дмитрий Иванович
Доктор экономических наук, руководитель Департамента «Налогов и налогового администрирования» Финансового университета при Правительстве РФ, ректор Института экономики и антикризисного управления, профессор Департамента «Антикризисное управление и финансы» ИЭАУ.
Управляющий партнер по налоговой практике юридической фирмы ООО «Легикон-Право», член Президентского Совета Института профессиональных бухгалтеров Московского региона, член Президентского Совета Института профессиональных бухгалтеров и аудиторов центрально-черноземного региона, Председатель комитета по профессиональному образованию ИПБ МР, заместитель главного редактора журнала «Вестник профессиональных бухгалтеров»""")
        
        assert result is not None
        assert result["start_time"] == "2024-09-20T10:00:00"
        assert result["end_time"] == "2024-09-20T18:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_tax_planning_webinar(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Как легально и выгодно вывести деньги из бизнеса — адаптируем налоговые решения к изменениям с 2025 года.
When
Friday, December 06, 2024
09:00 – 17:00 MSK (8 hours)
Where
https://my.mts-link.ru/j/119945689/625598312/f76edf9a32f370a9bfac3fa09ce8988e
Who
business.tinkoff@vvzvlad.xyz, invitation@webinar.ru
Notes
Сроки проведения:
06 декабря 2024 г. (с 9:00 до 17:00 по Москве)
Спикер: Кузьминых Артем Евгеньевич – управляющий партнёр компании Кузьминых и партнёры, консультант по налоговому планированию и построению холдинговых структур, аттестованный налоговый консультант (Ассоциация налоговых консультантов), преподаватель Национального исследовательского университета «Высшая школа экономики» (НИУ ВШЭ).""")
        
        assert result is not None
        assert result["start_time"] == "2024-12-06T09:00:00"
        assert result["end_time"] == "2024-12-06T17:00:00"
        assert result["location"] == "https://my.mts-link.ru/j/119945689/625598312/f76edf9a32f370a9bfac3fa09ce8988e"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_procurement_webinar(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 3, 15, 12, 0)
        result = await llm_instance.parse_calendar_event("""Новеллы в 44-ФЗ и 223-ФЗ для заказчиков и поставщиков. Изменения в национальном режиме и квотировании с 2025г. Новые алгоритмы электронных закупок.
19 декабря
09:00 (UTC +03) 
Принять приглашение 
Сроки проведения:
19–20 декабря 2024 г.
(с 9:00 до 17:00 по Москве)""")
        
        assert result is not None
        assert result["start_time"] == "2024-12-19T09:00:00"
        assert result["end_time"] == "2024-12-20T17:00:00"
        assert result["result"] is True
        assert result["comment"] is None

@pytest.mark.asyncio
async def test_parse_calendar_event_incomplete_phone(llm_instance):
    with patch('src.llm.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 3, 19, 12, 0)
        result = await llm_instance.parse_calendar_event("""Добрый день. Запись получили на субботу на 15:30. Напишите пожалуйста моб номер, Вы в записи вписали 892 632-73-65, последняя цифра не прошла.""")
        
        assert result is not None
        assert result["start_time"] == "2025-03-22T15:30:00"
        assert result["end_time"] == "2025-03-22T16:30:00"
        assert result["result"] is True
        assert result["comment"] is None
