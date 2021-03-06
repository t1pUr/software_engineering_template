# Аналіз тональності тексту за допомогою нейромереж

<br>Уявіть собі, що у вас є якась частина тексту, який ви ще не прочитали, але ви хочете знати, який він містить в собі настрій: радість, сум чи гнів?
В даному випадку ми будем класифікувати нашу задачу на 2 основні типи емоцій: ***позитині*** та ***негативні***.</br>
<br>Є багато способів вирішувати таке завдання. Один з них - це **згорткові нейронні мережі** (Convolutional Neural Networks). 
Ці нейронні мережі спочатку були розроблені для обробки зображень, однак вони успішно справляються з вирішенням завдань у сфері автоматичної обробки текстів. 
В даній роботі ми розглянемо бінарний аналіз тональності текстів українською за допомогою згорткової нейронної мережі,
для якої векторні представлення слів були сформовані на основі навченої **Word2Vec** моделі.</br>
![Sentiment_analysis_main](https://github.com/t1pUr/software_engineering_template/blob/master/src/images/Sentiment%20analysis%20main.gif)
<br>Джерело: https://habrastorage.org/webt/2u/l3/lw/2ul3lwsbyobovjnol2g_cbvrghi.gif</br>
<h2>
Застосування згорткових нейронних мереж (Convolutional Neural Networks) для задач NLP
</h2>
<br>
З самого початку згорткові нейронні мережі були призначені для розпізнавання та обробки зоображень. Через те, що вони влаштовані на зразок зорової кори головного мозку - тобто вміють концентруватися на невеликій області і виділяти в ній важливі особливості, CNN досягли значного успіху не тільки при роботі з зображеннями, але і для задач нейролінгвістичного програмування (Natural Language Processing, NLP).
</br>

<br>Що ж таке згортка? Розберемо приклад, а саме візуалізацію від Стенфорда:</br>
![Sentiment_analysis_main](https://github.com/t1pUr/software_engineering_template/blob/master/src/images/CNN_example.gif)
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/642/8cf/505/6428cf505ac1e9e1cf462e1ec8fe9a68.gif</br>
<br>Вікно, яке ходить по великій матриці називається фільтром (в англомовному варіанті ***kernel***, ***filter*** або ***feature detector***, можна зустріти декілька варіантів цих термінів). Фільтр накладається на ділянку великої матриці і кожне значення перемножується з відповідним йому значенням фільтра (червоні цифри нижче і правіше чорних цифр основної матриці). Потім все що вийшло складається і виходить кінцеве ("відфільтроване") значення. Цих фільтрів існує безліч та вони можуть ходити декілька разів по одній матриці, оскільки один й той самий фільтр може звертати увагу на різні деталі.</br>
<br>Вікно ходить по великій матриці з якимось кроком, який по-англійськи називається ***stride***. Цей крок буває горизонтальний і вертикальний (хоча останній нам не знадобиться).</br>
<br>Ще один важливий концепт - **канал**. Каналами в зображеннях називаються відомі багатьом базові кольори, наприклад, якщо ми говоримо про просту і поширену схему колірного кодування RGB (Red - червоний, Green - зелений, Blue - блакитний), то там передбачається, що з цих трьох базових кольорів, шляхом їх змішування ми можемо отримати будь-який колір.</br>
![Sentiment_analysis_main](https://github.com/t1pUr/software_engineering_template/blob/master/src/images/channels.png| width=400)
<br>Джерело: https://neurohive.io/wp-content/uploads/2018/07/kernels.png</br>

<br>Отже, у нас є зображення, в ньому є канали і по ньому з потрібним кроком ходить наш фільтр, але що саме відбувається з каналами? З цими каналами ми робимо наступне - кожен фільтр (тобто матриця невеликого розміру) накладається на вихідну матрицю одночасно на всі три канали. Результати ж просто підсумовуються.</br>

<br>Також є ще один базовий шар у CNN. Це так званий ***pooling-шар***. Розглянемо цей шар на прикладі ***max-pooling***. Уявіть, що у вже відомому вам згортковому шарі матриця фільтра зафіксована і є одиничною (тобто множення на неї ніяк не впливає на вхідні дані). А замість підсумовування всіх результатів множення ми просто вибираємо максимальний елемент. Тобто ми виберемо з усього вікна піксель з найбільшою інтенсивністю. Це і є max-pooling. Звичайно, замість цієї функцій може бути й інша арифметична, або навіть складніша функція.</br>
![Sentiment_analysis_main](https://github.com/t1pUr/software_engineering_template/blob/master/src/images/max_pooling.png)
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/e38/ee6/f95/e38ee6f954b7c7cea8f768c77eaff301.png</br>
<br>Тепер трохи відступимо від цієї теми та перейдемо до ***word embededing***</br>
<h2>
Word embedding або векторне представлення слів
</h2>

<br>Звідки береться сама задача word embedding?
На жаль, поки що не існує єдиного терміна для цього поняття, тому ми будемо використовувати англомовний.
Сам по собі embedding - це зіставлення довільної сутності (наприклад, вузла в графі або шматочка картинки) деякого вектору.</br>
![Sentiment_analysis_main](https://github.com/t1pUr/software_engineering_template/blob/master/src/images/embedding.png)
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/3e8/12f/d16/3e812fd164a08f5e4f195000fecf988f.png</br> 

<br>Ось у нас є слова і є комп'ютер, який повинен з цими словами якось працювати. Питання - як комп'ютер буде працювати зі словами? Адже комп'ютер не вміє "читати". Перше, що приходить в голову - просто закодувати слова цифрами по порядку проходження в словнику. Ідея дуже продуктивна в своїй простоті - натуральний ряд нескінченний і можна пронумерувати всі слова, не боючись проблем.</br>

<br>Але у цієї ідеї є і істотний недолік: слова в словнику слідують в алфавітному порядку, і при додаванні слова потрібно перенумеровувати заново більшу частину слів. Але навіть це не є настільки важливим, а важливо те, що буквене написання слова ніяк не пов'язане з його змістом. Наприклад, слова "півень", "курка" і "курча" мають дуже мало спільного між собою і стоять в словнику далеко один від одного, хоча очевидно позначають самця, самку і дитинча одного виду птиці. Тобто ми можемо виділити два види близькості слів: лексичний і семантичний. Як ми бачимо на прикладі з куркою, ці близькості не обов'язково збігаються. Можна для наочності привести зворотний приклад лексично близьких, але семантично далеких слів - "зола" та "золото". Щоб отримати можливість представити семантичну близькість, було запропоновано використовувати **embedding**, тобто зіставити слова в якийсь вектор, що відображає його значення в "просторі смислів".</br>

<br>Найпростішим способом буде взяти вектор довжини нашого словника і поставити тільки одиницю в позиції, що відповідає номеру слова в словнику. Цей підхід називається **one-hot encoding (OHE)**. Але **OHE** все ще не має властивості семантичної близькості:</br>
<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/one_hot_encoding.png"  width="413" height="379">
<br>Джерело: http://neerc.ifmo.ru/wiki/images/4/49/One-hot-encoding.png</br> 

<br>Значення одного слова нам може бути і не так важливо, тому що мова (і усна, і письмова) складається з наборів слів, які ми називаємо текстами. Так що якщо ми захочемо якось уявити тексти, то ми візьмемо OHE-вектор кожного слова в тексті і складемо разом. Тобто на виході отримаємо просто підрахунок кількості різних слів у тексті в одному векторі. Такий підхід називається "мішок слів" (**bag of words**, BoW), тому що ми втрачаємо всю інформацію про взаємне розташування слів у тексті.</br>

<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/bag_of_words.jfif">
<br>Джерело: https://www.programmersought.com/images/947/0acb9279d17a1631bcfb154583cca443.JPEG</br>

<br>Реалізуємо нашу першу модель "мішка слів" на мові Python. Перше, що нам потрібно - це набір даних. Для цього можна взяти якусь статтю з Вікіпедії. Але спочатку почнемо з бібліотек, які нам знадобляться</br>
```python
import nltk
import bs4 as bs
import urllib.request
import re
```
<br>Ми будемо використовувати бібліотеку **Beautifulsoup4** для аналізу даних з Вікіпедії. Крім того, регулярний вираз бібліотеки мови Python, **re** буде використовуватися для деяких Preprocessing завдань по тексту</br>
```python
raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Python_(programming_language)')
raw_html = raw_html.read()
article_html = bs.BeautifulSoup(raw_html, 'lxml')
article_paragraphs = article_html.find_all('p')
article_text = ''

for para in article_paragraphs:
    article_text += para.text
```
<br>В наведеному вище фрагменті ми імпортуємо необроблений HTML-код для статті у Вікіпедії. В даному випадку це стаття про мову Python. З необробленого HTML ми фільтруємо текст в тексті абзацу. Нарешті, ми створюємо повний корпус, об'єднавши всі абзаци. Наступний крок - розбити корпус на окремі речення. Для цього скористаємося методом sent_tokenize з бібліотеки NLTK.</br>
```python
corpus = nltk.sent_tokenize(article_text)
```
<br>Далі приводимо текст у нижній регіст та видаляємо усі розділові знаки. Оскільки видалення знаків пунктуації може призвести до появи декількох порожніх прогалин, скорегуємо порожні місця за допомогою регулярного виразу.</br>
```python
for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])
```
<br>Тепер у нас є власний корпус. Наступним кроком є створення словника, що містить слова і відповідні їм частоти в корпусі.</br>
```python
bag_of_words = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in bag_of_words.keys():
            bag_of_words[token] = 1
        else:
            bag_of_words[token] += 1
print(bag_of_words)
```
<br>Ми створили словник з ім'ям bag_of_words. Потім ми перебираємо кожне речення в корпусі. Потім ми перебираємо кожне слово в реченні. Якщо слово не існує в bag_of_words словнику, ми додамо слово в якості ключа і встановимо значення слова як 1. В іншому випадку, якщо слово вже існує в словнику, ми просто збільшимо лічильник ключів на 1. Далі створюємо список з словом та його частоти й сортуємо у порядку спадання.</br>
```python
pairs = []
list_of_words = bag_of_words.keys()
for i in list_of_words:
    pairs.append([i, bag_of_words[i]])

sorted_bag_of_words = reversed(sorted(pairs, key=lambda sort: sort[1]))
for i in sorted_bag_of_words:
    print(f"{i[0]} - {i[1]}", end=";\t")
```
<h3>Result</h3>

```
python - 150;	the - 144;	and - 131;	a - 109;	to - 84;	in - 83;	of - 83;	is - 78;	as - 57;	for - 53;	it - 33;	with - 33;	language - 32;	are - 31;	s - 29;	c - 27;	that - 26;	3 - 26;	programming - 25;	2 - 24;	or - 21;	languages - 20;	not - 20;	has - 19;	0 - 19;	its - 19;	be - 18;	by - 18;	such - 18;	was - 18;	used - 17;	from - 16;	b - 15;	can - 15;	standard - 15;	code - 15;	many - 14;	which - 14;	object - 14;	an - 14;	also - 13;	other - 13;	on - 13;	including - 13;	most - 12;	1 - 11;	5 - 11;	this - 11;	at - 11;	library - 11;	use - 11;	type - 10;	like - 10;	than - 10;	uses - 10;	development - 9;	but - 9;	time - 9;	some - 9;	support - 9;	example - 8;	have - 8;	cpython - 8;	reference - 8;	features - 8;	rossum - 8;	van - 8;	often - 8;	round - 7;	division - 7;	classes - 7;	java - 7;	class - 7;	statements - 7;	written - 7;	syntax - 7;	applications - 7;	expressions - 7;	program - 7;	web - 7;	include - 7;	implementation - 7;	one - 7;	7 - 7;	released - 7;	large - 7;	design - 7;	been - 6;	scripting - 6;	g - 6;	e - 6;	style - 6;	there - 6;	modules - 6;	name - 6;	supported - 6;	his - 6;	major - 6;	version - 6;	new - 6;	9 - 6;	oriented - 6;	negative - 5;	instance - 5;	expression - 5;	before - 5;	where - 5;	monty - 5;	software - 5;	while - 5;	into - 5;	functionality - 5;	pep - 5;	variable - 5;	versions - 5;	all - 5;	only - 5;	were - 5;	then - 5;	since - 5;	developers - 5;	core - 5;	community - 5;	guido - 5;	typed - 5;	indentation - 5;	philosophy - 5;	linux - 4;	libraries - 4;	computer - 4;	installer - 4;	programs - 4;	package - 4;	processing - 4;	third - 4;	several - 4;	precision - 4;	arbitrary - 4;	mathematics - 4;	provides - 4;	integer - 4;	numbers - 4;	positive - 4;	both - 4;	operator - 4;	being - 4;	part - 4;	information - 4;	however - 4;	data - 4;	if - 4;	available - 4;	contrast - 4;	means - 4;	rather - 4;	includes - 4;	tools - 4;	functions - 4;	method - 4;	typing - 4;	extensions - 4;	methods - 4;	issues - 4;	8 - 4;	will - 4;	council - 4;	steering - 4;	project - 4;	popular - 4;	2020 - 4;	system - 4;	list - 4;	first - 4;	functional - 4;	garbage - 4;	well - 4;	windows - 3;	ide - 3;	based - 3;	implementations - 3;	documentation - 3;	interface - 3;	server - 3;	user - 3;	commonly - 3;	numerical - 3;	scientific - 3;	party - 3;	decimal - 3;	equation - 3;	rounding - 3;	4 - 3;	operators - 3;	arithmetic - 3;	they - 3;	instances - 3;	types - 3;	allows - 3;	number - 3;	operations - 3;	self - 3;	sugar - 3;	between - 3;	function - 3;	may - 3;	statement - 3;	assignment - 3;	any - 3;	related - 3;	api - 3;	would - 3;	considered - 3;	do - 3;	their - 3;	less - 3;	interpreter - 3;	two - 3;	generator - 3;	during - 3;	names - 3;	binding - 3;	dynamic - 3;	security - 3;	more - 3;	easily - 3;	out - 3;	end - 3;	releases - 3;	x - 3;	now - 3;	life - 3;	when - 3;	until - 3;	does - 3;	using - 3;	comprehensions - 3;	abc - 3;	late - 3;	multiple - 3;	dynamically - 3;	projects - 3;	level - 3;	distributions - 2;	adopted - 2;	3d - 2;	rich - 2;	algebra - 2;	complex - 2;	google - 2;	year - 2;	february - 2;	games - 2;	show - 2;	prefix - 2;	167 - 2;	references - 2;	release - 2;	three - 2;	running - 2;	originally - 2;	hosted - 2;	specific - 2;	154 - 2;	primary - 2;	enhancement - 2;	various - 2;	performance - 2;	source - 2;	developed - 2;	compilers - 2;	macos - 2;	systems - 2;	platforms - 2;	machine - 2;	computing - 2;	developing - 2;	sagemath - 2;	browser - 2;	state - 2;	add - 2;	line - 2;	command - 2;	packages - 2;	contains - 2;	repository - 2;	official - 2;	index - 2;	update - 2;	2021 - 2;	need - 2;	platform - 2;	test - 2;	gateway - 2;	covered - 2;	testing - 2;	unit - 2;	relational - 2;	manipulation - 2;	problems - 2;	frequently - 2;	native - 2;	further - 2;	numpy - 2;	module - 2;	compared - 2;	derived - 2;	even - 2;	interval - 2;	result - 2;	true - 2;	105 - 2;	over - 2;	results - 2;	point - 2;	floating - 2;	respectively - 2;	math - 2;	traditional - 2;	these - 2;	125 - 2;	floor - 2;	experimental - 2;	static - 2;	directly - 2;	whether - 2;	same - 2;	old - 2;	kinds - 2;	itself - 2;	own - 2;	them - 2;	string - 2;	defined - 2;	compile - 2;	checked - 2;	argument - 2;	objects - 2;	valid - 2;	equality - 2;	error - 2;	classic - 2;	conditional - 2;	so - 2;	ruby - 2;	found - 2;	similar - 2;	through - 2;	passed - 2;	generators - 2;	better - 2;	values - 2;	contain - 2;	each - 2;	given - 2;	variables - 2;	others - 2;	among - 2;	four - 2;	meaning - 2;	t - 2;	doesn - 2;	side - 2;	feature - 2;	semantic - 2;	structure - 2;	visual - 2;	certain - 2;	special - 2;	syntactic - 2;	after - 2;	blocks - 2;	delimit - 2;	brackets - 2;	curly - 2;	keywords - 2;	referred - 2;	those - 2;	rough - 2;	natural - 2;	say - 2;	range - 2;	wide - 2;	pythonic - 2;	common - 2;	bar - 2;	foo - 2;	instead - 2;	eggs - 2;	spam - 2;	refer - 2;	examples - 2;	group - 2;	comedy - 2;	british - 2;	just - 2;	pypy - 2;	important - 2;	speed - 2;	parts - 2;	critical - 2;	non - 2;	optimization - 2;	culture - 2;	something - 2;	way - 2;	coding - 2;	choice - 2;	vision - 2;	interfaces - 2;	programmable - 2;	adding - 2;	extensible - 2;	68 - 2;	document - 2;	lisp - 2;	management - 2;	memory - 2;	via - 2;	metaprogramming - 2;	execution - 2;	possible - 2;	had - 2;	later - 2;	patches - 2;	existing - 2;	set - 2;	6 - 2;	collector - 2;	detecting - 2;	cycle - 2;	january - 2;	five - 2;	member - 2;	term - 2;	long - 2;	he - 2;	2018 - 2;	12 - 2;	lead - 2;	december - 2;	operating - 2;	38 - 2;	consistently - 2;	much - 2;	compatible - 2;	backward - 2;	completely - 2;	revision - 2;	2008 - 2;	counting - 2;	2000 - 2;	successor - 2;	1980s - 2;	began - 2;	due - 2;	particularly - 2;	structured - 2;	paradigms - 2;	supports - 2;	small - 2;	programmers - 2;	help - 2;	approach - 2;	readability - 2;	general - 2;	high - 2;	217 - 1;	swift - 1;	216 - 1;	erlang - 1;	215 - 1;	tcl - 1;	change - 1;	surrounding - 1;	rationale - 1;	describing - 1;	requiring - 1;	practice - 1;	emulated - 1;	practices - 1;	influenced - 1;	2013 - 1;	205 - 1;	provider - 1;	replace - 1;	intends - 1;	libreoffice - 1;	main - 1;	board - 1;	single - 1;	pi - 1;	raspberry - 1;	204 - 1;	labs - 1;	xo - 1;	child - 1;	per - 1;	laptop - 1;	203 - 1;	202 - 1;	exploit - 1;	industry - 1;	extensively - 1;	portage - 1;	gentoo - 1;	anaconda - 1;	fedora - 1;	hat - 1;	red - 1;	ubiquity - 1;	ubuntu - 1;	installers - 1;	terminal - 1;	openbsd - 1;	netbsd - 1;	freebsd - 1;	amigaos - 1;	201 - 1;	ships - 1;	component - 1;	200 - 1;	go - 1;	engine - 1;	app - 1;	199 - 1;	198 - 1;	video - 1;	197 - 1;	arcgis - 1;	scripts - 1;	writing - 1;	best - 1;	promotes - 1;	esri - 1;	containers - 1;	structures - 1;	printer - 1;	pretty - 1;	debugger - 1;	gnu - 1;	capella - 1;	scorewriter - 1;	notation - 1;	musical - 1;	196 - 1;	pro - 1;	shop - 1;	paint - 1;	scribus - 1;	inkscape - 1;	195 - 1;	gimp - 1;	imaging - 1;	2d - 1;	nuke - 1;	compositor - 1;	effects - 1;	softimage - 1;	motionbuilder - 1;	modo - 1;	maya - 1;	houdini - 1;	lightwave - 1;	4d - 1;	cinema - 1;	blender - 1;	max - 1;	3ds - 1;	animation - 1;	freecad - 1;	modeler - 1;	parametric - 1;	abaqus - 1;	element - 1;	finite - 1;	products - 1;	embedded - 1;	successfully - 1;	194 - 1;	text - 1;	simple - 1;	architecture - 1;	modular - 1;	193 - 1;	192 - 1;	191 - 1;	190 - 1;	learn - 1;	scikit - 1;	pytorch - 1;	keras - 1;	tensorflow - 1;	learning - 1;	intelligence - 1;	artificial - 1;	189 - 1;	image - 1;	bindings - 1;	opencv - 1;	188 - 1;	calculus - 1;	theory - 1;	combinatorics - 1;	aspects - 1;	covers - 1;	notebook - 1;	domain - 1;	providing - 1;	astropy - 1;	biopython - 1;	specialized - 1;	187 - 1;	186 - 1;	effective - 1;	allow - 1;	matplotlib - 1;	scipy - 1;	dropbox - 1;	computers - 1;	communications - 1;	framework - 1;	twisted - 1;	database - 1;	mapper - 1;	sqlalchemy - 1;	ajax - 1;	client - 1;	develop - 1;	ironpython - 1;	pyjs - 1;	maintenance - 1;	zope - 1;	bottle - 1;	flask - 1;	tornado - 1;	web2py - 1;	turbogears - 1;	pyramid - 1;	pylons - 1;	django - 1;	frameworks - 1;	facilitate - 1;	evolved - 1;	185 - 1;	apache - 1;	mod_wsgi - 1;	serve - 1;	184 - 1;	mostly - 1;	reddit - 1;	site - 1;	networking - 1;	news - 1;	social - 1;	183 - 1;	ita - 1;	182 - 1;	ilm - 1;	entities - 1;	smaller - 1;	181 - 1;	spotify - 1;	180 - 1;	instagram - 1;	amazon - 1;	179 - 1;	facebook - 1;	178 - 1;	nasa - 1;	177 - 1;	cern - 1;	176 - 1;	yahoo - 1;	175 - 1;	wikipedia - 1;	organizations - 1;	174 - 1;	worse - 1;	consumption - 1;	determined - 1;	dictionary - 1;	search - 1;	involving - 1;	conventional - 1;	productive - 1;	study - 1;	empirical - 1;	173 - 1;	172 - 1;	times - 1;	2010 - 1;	2007 - 1;	ratings - 1;	rise - 1;	highest - 1;	selected - 1;	171 - 1;	behind - 1;	tiobe - 1;	ten - 1;	top - 1;	ranked - 1;	2003 - 1;	gtk - 1;	qt - 1;	bind - 1;	pygtk - 1;	pyqt - 1;	create - 1;	sdl - 1;	pygame - 1;	py - 1;	170 - 1;	169 - 1;	routines - 1;	168 - 1;	literature - 1;	metasyntactic - 1;	appear - 1;	enjoyed - 1;	creator - 1;	whom - 1;	166 - 1;	graphviz - 1;	doxygen - 1;	forks - 1;	pdoc - 1;	sphinx - 1;	pydoc - 1;	generate - 1;	165 - 1;	matching - 1;	pattern - 1;	plans - 1;	164 - 1;	modified - 1;	163 - 1;	removed - 1;	wstr - 1;	deprecates - 1;	10 - 1;	pythons - 1;	pyladies - 1;	programmes - 1;	mentoring - 1;	pycon - 1;	conference - 1;	academic - 1;	162 - 1;	suite - 1;	monitors - 1;	team - 1;	ready - 1;	delayed - 1;	schedule - 1;	although - 1;	final - 1;	previews - 1;	candidates - 1;	beta - 1;	alpha - 1;	incremented - 1;	distinguished - 1;	come - 1;	public - 1;	157 - 1;	2017 - 1;	github - 1;	moved - 1;	mercurial - 1;	place - 1;	took - 1;	156 - 1;	org - 1;	bugs - 1;	tracker - 1;	bug - 1;	roundup - 1;	discussed - 1;	forum - 1;	dev - 1;	mailing - 1;	corresponds - 1;	commented - 1;	reviewed - 1;	peps - 1;	outstanding - 1;	155 - 1;	decisions - 1;	documenting - 1;	input - 1;	collecting - 1;	proposing - 1;	mechanism - 1;	process - 1;	proposal - 1;	largely - 1;	conducted - 1;	153 - 1;	game - 1;	benchmarks - 1;	benchmarked - 1;	152 - 1;	13 - 1;	euroscipy - 1;	presented - 1;	workload - 1;	combinatorial - 1;	comparison - 1;	subset - 1;	restricted - 1;	unrestricted - 1;	either - 1;	unsupported - 1;	lot - 1;	dropped - 1;	130 - 1;	solaris - 1;	os - 1;	frame - 1;	129 - 1;	priorities - 1;	earliest - 1;	portability - 1;	128 - 1;	vms - 1;	unofficial - 1;	macs - 1;	m1 - 1;	apple - 1;	unix - 1;	modern - 1;	xp - 1;	127 - 1;	126 - 1;	install - 1;	fails - 1;	deliberately - 1;	starting - 1;	mixture - 1;	distributed - 1;	virtual - 1;	executed - 1;	124 - 1;	bytecode - 1;	intermediate - 1;	compiles - 1;	123 - 1;	122 - 1;	c11 - 1;	implemented - 1;	older - 1;	limited - 1;	121 - 1;	120 - 1;	outdated - 1;	c99 - 1;	select - 1;	c89 - 1;	meeting - 1;	119 - 1;	emphasizing - 1;	commercial - 1;	canopy - 1;	environment - 1;	hosting - 1;	pythonanywhere - 1;	science - 1;	intended - 1;	ides - 1;	environments - 1;	integrated - 1;	desktop - 1;	highlighting - 1;	retention - 1;	session - 1;	completion - 1;	auto - 1;	improved - 1;	abilities - 1;	ipython - 1;	idle - 1;	shells - 1;	immediately - 1;	receives - 1;	sequentially - 1;	enters - 1;	permitting - 1;	repl - 1;	loop - 1;	print - 1;	eval - 1;	read - 1;	118 - 1;	000 - 1;	290 - 1;	pypi - 1;	march - 1;	variant - 1;	rewriting - 1;	altering - 1;	few - 1;	cross - 1;	because - 1;	suites - 1;	internal - 1;	specified - 1;	117 - 1;	333 - 1;	follows - 1;	wsgiref - 1;	wsgi - 1;	specifications - 1;	regular - 1;	manipulating - 1;	116 - 1;	decimals - 1;	pseudorandom - 1;	generating - 1;	databases - 1;	connecting - 1;	graphical - 1;	creating - 1;	http - 1;	mime - 1;	protocols - 1;	formats - 1;	facing - 1;	internet - 1;	tasks - 1;	suited - 1;	115 - 1;	strengths - 1;	greatest - 1;	cited - 1;	factorial - 1;	calculate - 1;	world - 1;	hello - 1;	114 - 1;	113 - 1;	aid - 1;	capabilities - 1;	extends - 1;	extensive - 1;	112 - 1;	rational - 1;	fractions - 1;	fraction - 1;	111 - 1;	modes - 1;	pre - 1;	110 - 1;	resulting - 1;	evaluate - 1;	differently - 1;	interpret - 1;	109 - 1;	tests - 1;	consistent - 1;	manner - 1;	relations - 1;	boolean - 1;	108 - 1;	zero - 1;	away - 1;	107 - 1;	produce - 1;	breaking - 1;	tie - 1;	nearest - 1;	float - 1;	106 - 1;	lie - 1;	open - 1;	half - 1;	expected - 1;	validity - 1;	maintaining - 1;	always - 1;	consistency - 1;	adds - 1;	different - 1;	though - 1;	infinity - 1;	towards - 1;	simply - 1;	terms - 1;	significantly - 1;	changed - 1;	behavior - 1;	produces - 1;	integers - 1;	represent - 1;	unary - 1;	infix - 1;	rules - 1;	precedence - 1;	work - 1;	104 - 1;	multiply - 1;	matrix - 1;	exponentiation - 1;	remainder - 1;	operation - 1;	modulo - 1;	symbols - 1;	usual - 1;	accessible - 1;	414 - 1;	100 - 1;	checking - 1;	mypy - 1;	named - 1;	checker - 1;	optional - 1;	default - 1;	specifying - 1;	99 - 1;	gradual - 1;	plan - 1;	eliminated - 1;	onwards - 1;	inherit - 1;	indirectly - 1;	inherited - 1;	difference - 1;	styles - 1;	98 - 1;	reflection - 1;	allowing - 1;	metaclass - 1;	eggsclass - 1;	spamclass - 1;	calling - 1;	constructed - 1;	define - 1;	sense - 1;	make - 1;	attempting - 1;	silently - 1;	forbidding - 1;	strongly - 1;	despite - 1;	suitable - 1;	signifying - 1;	fail - 1;	constraints - 1;	untyped - 1;	duck - 1;	97 - 1;	objective - 1;	implicit - 1;	access - 1;	parameter - 1;	explicit - 1;	normal - 1;	attached - 1;	causes - 1;	unintended - 1;	probably - 1;	syntactically - 1;	conditions - 1;	mistaking - 1;	avoiding - 1;	advantage - 1;	form - 1;	case - 1;	particular - 1;	lambda - 1;	duplicating - 1;	leads - 1;	scheme - 1;	enforced - 1;	rigidly - 1;	distinction - 1;	86 - 1;	levels - 1;	stack - 1;	back - 1;	pass - 1;	unidirectionally - 1;	iterators - 1;	lazy - 1;	85 - 1;	extending - 1;	provided - 1;	coroutine - 1;	84 - 1;	83 - 1;	never - 1;	according - 1;	continuations - 1;	call - 1;	tail - 1;	statically - 1;	contrasted - 1;	associated - 1;	fixed - 1;	holder - 1;	generic - 1;	rebound - 1;	subsequently - 1;	allocated - 1;	separate - 1;	operates - 1;	80 - 1;	spaces - 1;	size - 1;	indent - 1;	recommended - 1;	share - 1;	rule - 1;	off - 1;	termed - 1;	sometimes - 1;	represents - 1;	accurately - 1;	thus - 1;	79 - 1;	block - 1;	current - 1;	signifies - 1;	decrease - 1;	comes - 1;	increase - 1;	whitespace - 1;	78 - 1;	pascal - 1;	cases - 1;	exceptions - 1;	fewer - 1;	ever - 1;	rarely - 1;	allowed - 1;	semicolons - 1;	unlike - 1;	punctuation - 1;	english - 1;	uncluttered - 1;	visually - 1;	formatting - 1;	readable - 1;	meant - 1;	77 - 1;	76 - 1;	pythonistas - 1;	experienced - 1;	knowledgeable - 1;	especially - 1;	admirers - 1;	users - 1;	75 - 1;	74 - 1;	unpythonic - 1;	called - 1;	another - 1;	transcription - 1;	reads - 1;	understand - 1;	difficult - 1;	emphasis - 1;	minimalist - 1;	conforms - 1;	fluency - 1;	shows - 1;	idioms - 1;	meanings - 1;	neologism - 1;	73 - 1;	72 - 1;	sketch - 1;	famous - 1;	materials - 1;	tutorials - 1;	approaches - 1;	playful - 1;	occasionally - 1;	71 - 1;	tribute - 1;	reflected - 1;	fun - 1;	keeping - 1;	goal - 1;	calls - 1;	direct - 1;	makes - 1;	script - 1;	translates - 1;	cython - 1;	compiler - 1;	extension - 1;	move - 1;	programmer - 1;	70 - 1;	clarity - 1;	cost - 1;	increases - 1;	marginal - 1;	offer - 1;	reject - 1;	premature - 1;	avoid - 1;	strive - 1;	69 - 1;	compliment - 1;	clever - 1;	describe - 1;	writes - 1;	author - 1;	book - 1;	foundation - 1;	fellow - 1;	martelli - 1;	alex - 1;	obvious - 1;	preferably - 1;	should - 1;	embraces - 1;	motto - 1;	perl - 1;	methodology - 1;	giving - 1;	grammar - 1;	cluttered - 1;	simpler - 1;	strives - 1;	opposite - 1;	espoused - 1;	frustrations - 1;	stemmed - 1;	made - 1;	modularity - 1;	compact - 1;	highly - 1;	designed - 1;	built - 1;	having - 1;	aphorisms - 1;	20 - 1;	zen - 1;	summarized - 1;	67 - 1;	ml - 1;	haskell - 1;	borrowed - 1;	implement - 1;	functools - 1;	itertools - 1;	66 - 1;	sets - 1;	dictionaries - 1;	mapandreduce - 1;	filter - 1;	tradition - 1;	offers - 1;	binds - 1;	resolution - 1;	65 - 1;	combination - 1;	64 - 1;	logic - 1;	63 - 1;	62 - 1;	contract - 1;	61 - 1;	magic - 1;	metaobjects - 1;	60 - 1;	aspect - 1;	fully - 1;	paradigm - 1;	multi - 1;	59 - 1;	poisoning - 1;	cache - 1;	58 - 1;	remote - 1;	leading - 1;	57 - 1;	56 - 1;	expedited - 1;	55 - 1;	54 - 1;	53 - 1;	improvements - 1;	no - 1;	52 - 1;	51 - 1;	ported - 1;	forward - 1;	could - 1;	body - 1;	concern - 1;	postponed - 1;	2015 - 1;	initially - 1;	date - 1;	50 - 1;	translation - 1;	partially - 1;	least - 1;	automates - 1;	utility - 1;	2to3 - 1;	series - 1;	49 - 1;	backported - 1;	48 - 1;	47 - 1;	unicode - 1;	october - 1;	16 - 1;	46 - 1;	nomination - 1;	withdrawn - 1;	45 - 1;	willing - 1;	carol - 1;	warsaw - 1;	barry - 1;	coghlan - 1;	nick - 1;	cannon - 1;	brett - 1;	elected - 1;	active - 1;	2019 - 1;	44 - 1;	43 - 1;	42 - 1;	person - 1;	leadership - 1;	shares - 1;	41 - 1;	maker - 1;	decision - 1;	chief - 1;	commitment - 1;	reflect - 1;	him - 1;	upon - 1;	bestowed - 1;	title - 1;	dictator - 1;	benevolent - 1;	responsibilities - 1;	vacation - 1;	permanent - 1;	announced - 1;	july - 1;	developer - 1;	responsibility - 1;	sole - 1;	shouldered - 1;	40 - 1;	1989 - 1;	amoeba - 1;	interfacing - 1;	handling - 1;	exception - 1;	capable - 1;	39 - 1;	setl - 1;	inspired - 1;	netherlands - 1;	cwi - 1;	informatica - 1;	wiskunde - 1;	centrum - 1;	conceived - 1;	37 - 1;	36 - 1;	35 - 1;	34 - 1;	33 - 1;	ranks - 1;	unmodified - 1;	run - 1;	32 - 1;	18 - 1;	discontinued - 1;	collection - 1;	introduced - 1;	31 - 1;	1991 - 1;	working - 1;	30 - 1;	comprehensive - 1;	included - 1;	batteries - 1;	described - 1;	procedural - 1;	collected - 1;	29 - 1;	scale - 1;	logical - 1;	clear - 1;	write - 1;	aim - 1;	constructs - 1;	significant - 1;	notable - 1;	emphasizes - 1;	purpose - 1;	interpreted - 1;
```
<br>Повний код доступний за лінком: https://github.com/t1pUr/software_engineering_template/blob/master/src/bag_of_words.py</br>

<br>Всі ці вище перераховані підходи є корисними в тих випадках, де розмір текстів невеликий та обмежений словник. Але з приходом в наше життя інтернета все стало одночасно і складніше і простіше: в доступі з'явилося безліч текстів. З цим треба було щось робити, а раніше відомі моделі не могли впоратися з таким обсягом текстів. І тоді, був запропонований вихід за принципом "той, хто нам заважає, той нам допоможе!" А саме, в 2013 році тоді мало кому відомий чеський аспірант Томаш Міколов запропонував свій підхід до word embedding, який він назвав ***Word2vec***. Його підхід заснований на іншій важливій гіпотезі, яку в науці прийнято називати ***гіпотезою локальності*** - "слова, які зустрічаються в однакових оточеннях, мають близькі значення". Близькість в даному випадку розуміється дуже широко, як те, що поруч можуть стояти тільки поєднуються слова. Наприклад, для нас звично словосполучення "заводний будильник". А сказати "заводний апельсин" ми не можемо, оскільки ці слова не поєднуються. Таким чином, Томаш Міколов запропонував новий підхід, який не страждав від великих обсягів інформації, а навпаки вигравав.</br>
<br>Модель дуже проста - ми будемо передбачати ймовірність слова по його контексту. Тобто ми будемо вчити такі вектори слів, щоб ймовірність, яка належить моделі слів була близька до ймовірності зустріти це слово в реальному тексті.</br>

<br>Варто відзначити: хоча в модель не закладено явно ніякої семантики, а тільки статистичні властивості корпусів текстів, виявляється, що натренований модель word2vec може вловлювати деякі семантичні властивості слів. Спочатку розглянемо детальніше як працює ***Word2vec***.</br>
<br>Наприклад, охарактерихуємо якусь людину по шкалі від -1 до 1 екстраверт вона чи інтроверт? (-1 - максимально інтровертний тип, а 1 - максимально екстравертний). 
Припустимо, це число -0.4</br>

<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/extraversion.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/b39/e23/ce1/b39e23ce1c036b11763e3c45c3659a3e.png</br>

<br>Чесно кажучи, тільки по 1 характеристиці дкже важко дізнатися про людину, тому додамо ще одну характеристику, не називаючи її, щоб визначити характеристику людини в цілому</br>

<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/character2.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/2aa/ab2/ebc/2aaab2ebc1ff30f1fd832e5cf5bf9cb1.png</br>

<br>Тепер можна сказати, що цей вектор характеризує якусь особистість. Це корисно, якщо порівнювати з іншими людьми. Припустимо, я водій червоного автобуса, яких з двох людей на графіку більш схожий на мене?</br>

<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/character_bus.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/de5/380/b84/de5380b84dc9fec4bb8b52ebe6519e15.png</br>

<br>При роботі з векторами схожість зазвичай обчислюється за геометричним коефіцієнтом:</br>

<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/calculation_similarity.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/640/e59/7dd/640e597dd741a28bcec986454633e31d.png</br>

<br>Як бачимо, людина №1 більше схожа на мене за характером. Вектори в одному напрямку (довжина також важлива) дають більший геометричний коефіцієнт. Насправді кількість таких вимірів у ***Word2vec*** набагато більша, а саме 50. Проблема з такою кількістю вимірів в тому, що вже не вийде накреслити стрілки в 2D просторі. Це загальна проблема в машинному навчанні, де часто доводиться працювати в багатовимірному просторі. Добре, що геометричний коефіцієнт працює з будь-якою кількістю вимірів, тому чим більше їх, тим точніше буде результат порівнювання</br>
<br>Отже, тут є 2 головні ідеї:
* Об'єкти можна представити у вигляді числових векторів, що відмінно підходить для машин.
* Можна легко визначити схожість векторів.</br>

<br>Перейдемо до векторних уявлень слів, отриманих в результаті навчання і подивимося на їх цікаві властивості. Ось вкладення для слова «король» (вектор GloVe, навчений на Вікіпедії):</br>

```
[ 0.50451 , 0.68607 , -0.59517 , -0.022801, 0.60046 , -0.13498 , -0.08813 , 0.47377 , -0.61798 , -0.31012 , -0.076666, 1.493 , -0.034189, -0.98173 , 0.68229 , 0.81722 , -0.51874 , -0.31503 , -0.55809 , 0.66421 , 0.1961 , -0.13495 , -0.11476 , -0.30344 , 0.41177 , -2.223 , -1.0756 , -1.0783 , -0.34354 , 0.33505 , 1.9927 , -0.04234 , -0.64319 , 0.71125 , 0.49159 , 0.16754 , 0.34344 , -0.25663 , -0.8523 , 0.1661 , 0.40102 , 1.1685 , -1.0137 , -0.21585 , -0.15155 , 0.78321 , -0.91241 , -1.6106 , -0.64426 , -0.51042 ]
```

<br>Ми бачимо список з 50 чисел, але по ним важко щось сказати. Давайте їх визуалізуємо, щоб порівняти з іншими векторами. Помістимо числа в один ряд та розфарбуємо кожну комірку (червоний ближче до 2, синій ближчий до -2, а білий - 0)</br>


<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/king.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/46f/7cb/1d5/46f7cb1d5adc32bd16368b2681ab26a4.png</br>

<br>Тепер відкинемо всі числа та просто порівняємо "короля" з іншими словами</br>


<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/king_man_woman.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/1c8/6b2/909/1c86b290963e8a42b375cb6a71245185.png</br>

<br>Бачимо, що «чоловік» і «жінка» набагато ближче один до одного, ніж до «короля»? Це про щось говорить. Векторні уявлення захоплюють досить багато інформації, значення та асоціацій цих слів. Знамениті приклади, які показують неймовірні властивості вкладень, - поняття аналогій. Ми можемо додавати і віднімати вектори слів, отримуючи цікаві результати. Найвідоміший приклад - формула «король - чоловік + жінка». Візуалізуємо цю аналогію</br>


<img src="https://github.com/t1pUr/software_engineering_template/blob/master/src/images/queen.png">
<br>Джерело: https://habrastorage.org/getpro/habr/post_images/a19/84b/fea/a1984bfeab5a597c6fb6300f7d694901.png</br>

<br>Отриманий вектор від обчислення «король-чоловік + жінка» не зовсім дорівнює «королеві», але це найбільш близький результат з 400000 вкладень слів в наборі даних.</br>

<br>Розберемо модель ***Word2vec*** на основі датасету російськомовних Twitter-постів. Для початку ми підготуємо цей датасет, який складається з твітів з позитивними та негативними відтінками, зібраною Ю.Рубцовою. Насамперед, потрібно скачати корпус позитивних та негативних твітів в форматі .csv, які можна скачати за посиланням: http://study.mokoron.com (positive.csv та negative.csv відповідно) та імпортувати відповідні бібліотеки:</br>

```python
import nltk
import pandas as pd
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
```

<br>Після установки цих Python-бібліотек необхідно отримати стоп-слова від NLTK, далі зчитаємо файли за допомогою бібліотеки Pandas</br>

```python
nltk.download('stopwords')

df_pos = pd.read_csv("positive.csv", sep=";", header=None)
df_neg = pd.read_csv("negative.csv", sep=";", header=None)
```

<br>Обидва файли мають безліч стовпців, такі як ім'я автора поста, дата публікації і т.д., але нас цікавить тільки самі twitter-пости, також нас не цікавить позитивні ці твіти або негативні, тому ми просто об'єднаємо їх</br>

```python
df = df_pos.iloc[:, 3].append(df_neg.iloc[:, 3])
df = df.dropna().drop_duplicates()
```

<br>При роботі з NLP-завданнями потрібно нормалізувати дані. В даному випадку, ми проведемо лематизацію, тобто приведення слів до нормальної форми і видалимо стоп-слова, скориставшись Python-бібліотеками pymorphy2 і NLTK відповідно. Порядок дій такий:
1. позбудемося букв латинського алфавіту, чисел, знаків пунктуації та всіх символів, наприклад, символ @ зустрічається майже всюди;
2. розіб'ємо пост на токени;
3. проведемо лематизацію;
4. видалимо стоп-слова.</br>

```python
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]

            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None
```

<br>Щоб отримати датасет, залишається тільки застосувати функцію до об'єкта DataFrame</br>

```python
data = df.apply(lemmatize)
data = data.dropna()
```
<br>Підготувавши датасет, можемо навчити модель. Для цього скористаємося бібліотекою Gensim та ініціалізуємо модель Word2vec та отримаємо словник</br>

```python
w2v_model = Word2Vec(
    min_count=10,
    window=2,
    vector_size=300,
    negative=10,
    alpha=0.03,
    min_alpha=0.0007,
    sample=6e-5,
    sg=1)

w2v_model.build_vocab(data)
```

<br>Розпишемо кожен з аргументів:
* min_count - ігнорувати всі слова з частотою зустрічальності менше, ніж це значення.
* windоw - розмір контекстного вікна, про який говорили тут, позначає діапазон контексту.
* vector_size - розмір векторного уявлення слова (word embedding).
* negative - скільки неконтекстне слів враховувати в навчанні, використовуючи negative sampling.
* alpha - початковий learning_rate, який використовується в алгоритмі зворотного поширення помилки (Backpropogation).
* min_alpha - мінімальне значення learning_rate, на яке може опуститися в процесі навчання.
* sg - якщо 1, то використовується реалізація Skip-gram; якщо 0, то CBOW.</br>
<br>Навчаємо модель, використовуючи метод train:</br>

```python
w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
```

<br>Після того, як модель була навчена, можемо дивитися результати. Кожне слово представляється вектором, отже, їх можна порівнювати. Як інструмент порівняння в Gensim використовується косінусний коефіцієнт (Cosine similarity). У моделі Word2vec є як атрибут об'єкт **wv**, який і містить векторне подання слів (word embeddings). У цього об'єкта є методи для отримання заходів схожості слів. Наприклад, визначимо, які слова знаходяться найближче до слова "любить":</br>

```python
print(w2v_model.wv.most_similar(positive=["любить"]))
```

```
[('скучать', 0.4844067692756653), ('дорожить', 0.47420695424079895), ('ты', 0.46857950091362), ('я', 0.46783530712127686), ('шеннон', 0.46409815549850464), ('безответно', 0.4439948499202728), ('люблюий', 0.4425821900367737), ('хочть', 0.4420243203639984), ('хотеть', 0.4402197301387787), ('любиш', 0.4382258355617523)]
```
<br>Число після коми позначає косінусний коефіцієнт. Чим він більший, тим вище близькість слів. Можна помітити, що слова "дорожить", "скучать", "ты" найбільш близькі до слова "любить". Перевіримо ще одне слово:</br>

```python
print(w2v_model.wv.most_similar(positive=["мужчина"]))
```

```
[('женщина', 0.4826136529445648), ('парень', 0.3630697429180145), ('ускоряться', 0.3491373658180237), ('девушка', 0.34378883242607117), ('грешить', 0.34376007318496704), ('именинник', 0.3417075574398041), ('человек', 0.3379362225532532), ('имидж', 0.3342508375644684), ('стеснительный', 0.32812944054603577), ('любовник', 0.3266494870185852)]
```

<br>Вектори ще можна додавати та віднімати. Наприклад, "день"+"завтра" та "папа"+"брат"-"мама"</br>

```python
print(w2v_model.wv.most_similar(positive=["день", "завтра"]))
```
```
[('сегодня', 0.7138642072677612), ('твитотмафия', 0.5662438273429871), ('неделя', 0.5621627569198608), ('семестровый', 0.5446457862854004), ('табель', 0.5411765575408936), ('ехууа', 0.5380048155784607), ('денёчек', 0.5347149968147278), ('отсыпаться', 0.5312281250953674), ('утро', 0.5265368223190308), ('суббота', 0.5243238806724548)]
```

```python
print(w2v_model.wv.most_similar(positive=["папа", "брат"], negative=["мама"]))
```
```
[('сестра', 0.3372751474380493), ('братец', 0.2793600559234619), ('исин', 0.26645463705062866), ('придурок', 0.2650151550769806), ('младший', 0.26485374569892883), ('двоюродный', 0.2597059905529022), ('старинный', 0.2481071799993515), ('несчастие', 0.24800485372543335)]
```
<br>Цей код знаходиться у мене в репозиторії за лінком: https://github.com/t1pUr/software_engineering_template/blob/master/src/word2vec.py</br>
