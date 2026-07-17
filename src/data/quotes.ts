/**
 * Quote corpus for the about-page rotator.
 * 100 entries — EN 40, CN 40, others 20.
 * Author prefix "—" is preserved in `a`.
 */
export type QuoteCat =
	| "EN"
	| "CN"
	| "DE"
	| "LAT"
	| "FR"
	| "RU"
	| "ES"
	| "JP"
	| "IT";

export interface Quote {
	q: string;
	a: string;
	cat: QuoteCat;
}

// === English (40) ===
const EN: Quote[] = [
	{ q: `Talk is cheap. Show me the code.`, a: `— Linus Torvalds`, cat: `EN` },
	{ q: `Stay hungry, stay foolish.`, a: `— Steve Jobs`, cat: `EN` },
	{
		q: `I've seen things you people wouldn't believe... all those moments will be lost in time, like tears in rain.`,
		a: `— Blade Runner`,
		cat: `EN`,
	},
	{ q: `The code is the law.`, a: `— Lawrence Lessig`, cat: `EN` },
	{ q: `Information wants to be free.`, a: `— Stewart Brand`, cat: `EN` },
	{ q: `Everything is a file.`, a: `— Unix Philosophy`, cat: `EN` },
	{
		q: `In the beginning was the command line.`,
		a: `— Neal Stephenson`,
		cat: `EN`,
	},
	{ q: `Don't panic.`, a: `— Douglas Adams`, cat: `EN` },
	{
		q: `The best way to predict the future is to invent it.`,
		a: `— Alan Kay`,
		cat: `EN`,
	},
	{
		q: `There are only 10 types of people: those who understand binary, and those who don't.`,
		a: `— Anonymous`,
		cat: `EN`,
	},
	{ q: `Recursion: See Recursion.`, a: `— Developer Manual`, cat: `EN` },
	{ q: `There is no place like 127.0.0.1.`, a: `— Geek Proverb`, cat: `EN` },
	{
		q: `Software is eating the world.`,
		a: `— Marc Andreessen`,
		cat: `EN`,
	},
	{
		q: `To be, or not to be, that is the question.`,
		a: `— William Shakespeare`,
		cat: `EN`,
	},
	{
		q: `What's past is prologue.`,
		a: `— William Shakespeare`,
		cat: `EN`,
	},
	{
		q: `Success is not final, failure is not fatal.`,
		a: `— Winston Churchill`,
		cat: `EN`,
	},
	{
		q: `We are all in the gutter, but some of us are looking at the stars.`,
		a: `— Oscar Wilde`,
		cat: `EN`,
	},
	{
		q: `Not all those who wander are lost.`,
		a: `— J.R.R. Tolkien`,
		cat: `EN`,
	},
	{ q: `Stay close to the metal.`, a: `— Hacker Proverb`, cat: `EN` },
	{ q: `Hello World.`, a: `— Root`, cat: `EN` },
	{
		q: `I'd tell you a joke about UDP, but you might not get it.`,
		a: `— Network Joke`,
		cat: `EN`,
	},
	{
		q: `One man's constant is another man's variable.`,
		a: `— Alan Perlis`,
		cat: `EN`,
	},
	{
		q: `Hardware is where the puppet injects its soul.`,
		a: `— Ghost in the Shell`,
		cat: `EN`,
	},
	{
		q: `Move fast and break things.`,
		a: `— Mark Zuckerberg`,
		cat: `EN`,
	},
	{ q: `Keep it simple, stupid.`, a: `— Kelly Johnson`, cat: `EN` },
	{ q: `Live long and prosper.`, a: `— Spock`, cat: `EN` },
	{
		q: `Logic is the beginning of wisdom, not the end.`,
		a: `— Spock`,
		cat: `EN`,
	},
	{ q: `Winter is coming.`, a: `— George R.R. Martin`, cat: `EN` },
	{ q: `The cake is a lie.`, a: `— Portal`, cat: `EN` },
	{
		q: `Big Brother is watching you.`,
		a: `— George Orwell`,
		cat: `EN`,
	},
	{
		q: `Everything you can imagine is real.`,
		a: `— Pablo Picasso`,
		cat: `EN`,
	},
	{ q: `So it goes.`, a: `— Kurt Vonnegut`, cat: `EN` },
	{
		q: `The unexamined life is not worth living.`,
		a: `— Socrates`,
		cat: `EN`,
	},
	{
		q: `Computers are useless. They can only give you answers.`,
		a: `— Pablo Picasso`,
		cat: `EN`,
	},
	{
		q: `A room without books is like a body without a soul.`,
		a: `— Cicero`,
		cat: `EN`,
	},
	{
		q: `Freedom means the opportunity to be what we never thought we would be.`,
		a: `— Daniel J. Boorstin`,
		cat: `EN`,
	},
	{
		q: `The future is already here — it's just not very evenly distributed.`,
		a: `— William Gibson`,
		cat: `EN`,
	},
	{
		q: `All watched over by machines of loving grace.`,
		a: `— Richard Brautigan`,
		cat: `EN`,
	},
	{ q: `End of File. System Halt.`, a: `— Kernel`, cat: `EN` },
	{ q: `I think, therefore I am.`, a: `— Descartes`, cat: `EN` },
];

// === Chinese (40) ===
const CN: Quote[] = [
	{ q: `代码是人类写给未来的情书。`, a: `— 匿名`, cat: `CN` },
	{ q: `回首向来萧瑟处，归去，也无风雨也无晴。`, a: `— 苏轼`, cat: `CN` },
	{ q: `吹灭读书灯，一身都是月。`, a: `— 孙皓晖`, cat: `CN` },
	{
		q: `草在结它的种子，风在摇它的叶子，我们站着，不说话，就十分美好。`,
		a: `— 顾城`,
		cat: `CN`,
	},
	{ q: `林深时见鹿，海蓝时见鲸，梦醒时见你。`, a: `— 佚名`, cat: `CN` },
	{
		q: `如果你认识从前的我，那么你就会原谅现在的我。`,
		a: `— 张爱玲`,
		cat: `CN`,
	},
	{
		q: `我见青山多妩媚，料青山见我应如是。`,
		a: `— 辛弃疾`,
		cat: `CN`,
	},
	{ q: `落霞与孤鹜齐飞，秋水共长天一色。`, a: `— 王勃`, cat: `CN` },
	{ q: `大漠孤烟直，长河落日圆。`, a: `— 王维`, cat: `CN` },
	{ q: `愿你出走半生，归来仍是少年。`, a: `— 佚名`, cat: `CN` },
	{ q: `岁月失语，惟石能言。`, a: `— 马伯庸`, cat: `CN` },
	{ q: `世界之大，唯有沉默最震耳欲聋。`, a: `— 佚名`, cat: `CN` },
	{ q: `既然选择了远方，便只顾风雨兼程。`, a: `— 汪国真`, cat: `CN` },
	{ q: `所有的痛苦本质上都是对无能的愤怒。`, a: `— 王小波`, cat: `CN` },
	{ q: `博观而约取，厚积而薄发。`, a: `— 苏轼`, cat: `CN` },
	{ q: `人生如逆旅，我亦是行人。`, a: `— 苏轼`, cat: `CN` },
	{ q: `此情可待成追忆？只是当时已惘然。`, a: `— 李商隐`, cat: `CN` },
	{ q: `醉后不知天在水，满船清梦压星河。`, a: `— 唐温如`, cat: `CN` },
	{
		q: `每一个不曾起舞的日子，都是对生命的辜负。`,
		a: `— 尼采 (CN)`,
		cat: `CN`,
	},
	{ q: `此心安处是吾乡。`, a: `— 苏轼`, cat: `CN` },
	{
		q: `黑夜给了我黑夜的眼睛，我却用它寻找光明。`,
		a: `— 顾城`,
		cat: `CN`,
	},
	{ q: `心之所向，素履以往。`, a: `— 七堇年`, cat: `CN` },
	{ q: `欲买桂花同载酒，终不似，少年游。`, a: `— 刘过`, cat: `CN` },
	{
		q: `世界上最遥远的距离，是你在写 C++，我在写 HTML。`,
		a: `— 极客段子`,
		cat: `CN`,
	},
	{
		q: `如果你的生活出现了 Bug，记得先重启一下心情。`,
		a: `— 极客语录`,
		cat: `CN`,
	},
	{ q: `所谓自由，就是从 0 到 1 的勇气。`, a: `— 极客精神`, cat: `CN` },
	{
		q: `昔我往矣，杨柳依依。今我来思，雨雪霏霏。`,
		a: `— 《诗经》`,
		cat: `CN`,
	},
	{ q: `无可奈何花落去，似曾相识燕归来。`, a: `— 晏殊`, cat: `CN` },
	{ q: `海上生明月，天涯共此时。`, a: `— 张九龄`, cat: `CN` },
	{ q: `星垂平野阔，月涌大江流。`, a: `— 杜甫`, cat: `CN` },
	{
		q: `满地都是六便士，他却抬头看见了月亮。`,
		a: `— 毛姆 (CN)`,
		cat: `CN`,
	},
	{ q: `凡是过往，皆为序章。`, a: `— 莎士比亚 (CN)`, cat: `CN` },
	{ q: `天地不仁，以万物为刍狗。`, a: `— 老子`, cat: `CN` },
	{ q: `生如夏花之绚烂，死如秋叶之静美。`, a: `— 泰戈尔 (CN)`, cat: `CN` },
	{ q: `于无声处听惊雷。`, a: `— 鲁迅`, cat: `CN` },
	{ q: `所有的人都渴望光，但没有人愿意去燃烧。`, a: `— 佚名`, cat: `CN` },
	{ q: `静水流深。`, a: `— 谚语`, cat: `CN` },
	{ q: `青山一道同云雨，明月何曾是两乡。`, a: `— 王昌龄`, cat: `CN` },
	{
		q: `Silent communication is the highest protocol.`,
		a: `— Anonymous`,
		cat: `CN`,
	},
];

// === German (3) ===
const DE: Quote[] = [
	{
		q: `Was mich nicht umbringt, macht mich stärker.`,
		a: `— Friedrich Nietzsche`,
		cat: `DE`,
	},
	{
		q: `Wovon man nicht sprechen kann, darüber muss man schweigen.`,
		a: `— Ludwig Wittgenstein`,
		cat: `DE`,
	},
	{ q: `Du mußt dein Ändern leben.`, a: `— Rainer Maria Rilke`, cat: `DE` },
];

// === Latin (3) ===
const LAT: Quote[] = [
	{ q: `Cogito, ergo sum.`, a: `— René Descartes`, cat: `LAT` },
	{ q: `Amor Fati.`, a: `— Friedrich Nietzsche`, cat: `LAT` },
	{ q: `Per aspera ad astra.`, a: `— Seneca`, cat: `LAT` },
];

// === French (4) ===
const FR: Quote[] = [
	{
		q: `On ne voit bien qu'avec le cœur.`,
		a: `— Antoine de Saint-Exupéry`,
		cat: `FR`,
	},
	{
		q: `Au milieu de l'hiver, j'apprenais enfin qu'il y avait en moi un été invincible.`,
		a: `— Albert Camus`,
		cat: `FR`,
	},
	{
		q: `L'enfer, c'est les autres.`,
		a: `— Jean-Paul Sartre`,
		cat: `FR`,
	},
	{
		q: `Il n'y a qu'un héroïsme au monde : voir le monde tel qu'il est, et de l'aimer.`,
		a: `— Romain Rolland`,
		cat: `FR`,
	},
];

// === Russian (2) ===
const RU: Quote[] = [
	{
		q: `Если жизнь тебя обманет, Не печалься, не сердись!`,
		a: `— Александр Пушкин`,
		cat: `RU`,
	},
	{ q: `Красота спасет мир.`, a: `— Фёдор Достоевский`, cat: `RU` },
];

// === Spanish (2) ===
const ES: Quote[] = [
	{
		q: `La vida no es la que uno vivió, sino la que uno recuerda.`,
		a: `— García Márquez`,
		cat: `ES`,
	},
	{
		q: `Podrán cortar todas las flores, pero no podrán detener la primavera.`,
		a: `— Pablo Neruda`,
		cat: `ES`,
	},
];

// === Japanese (4) ===
const JP: Quote[] = [
	{ q: `月が綺麗ですね。`, a: `— 夏目 漱石`, cat: `JP` },
	{ q: `散る桜 残る桜も 散る桜。`, a: `— 良寛`, cat: `JP` },
	{ q: `古池や 蛙飛び込む 水の音。`, a: `— 松尾 芭蕉`, cat: `JP` },
	{ q: `雨ニモマケズ、風ニモマケズ。`, a: `— 宮沢 贤治`, cat: `JP` },
];

// === Italian (2) ===
const IT: Quote[] = [
	{
		q: `L'amor che move il sole e l'altre stelle.`,
		a: `— Dante Alighieri`,
		cat: `IT`,
	},
	{ q: `E pur si muove.`, a: `— Galileo Galilei`, cat: `IT` },
];

export const quotes: Quote[] = [...EN, ...CN, ...DE, ...LAT, ...FR, ...RU, ...ES, ...JP, ...IT];