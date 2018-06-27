import os

E = ['AllTogether', 'Baseball', 'Boy-Girl', 'CVS', 'C_chat', 'GameSale', 'GetMarry', 'Lifeismoney', 'LoL', 'MH', 'MLB', 'Mobilecomm', 'movie', 'MuscleBeach', 'NBA', 'SENIORHIGH', 'Stock', 'Tennis', 'Tos', 'WomenTalk']
e = ['span', 'gt', 'lt', 'class', 'amp']#, '`', 'ˊ', 'ﾟ', 'ˇ']

if __name__ == '__main__':
	for i in E:
		for j in os.listdir('./Training/{key}_cut/'.format(key=i)):
			a = []
			for k in open("./Training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().split(' / '):
				if k not in e and len(k) < 10: a.append(k)
			open("./Training/{key}_cut/{file}".format(key=i, file=j), 'w', encoding='utf-8').write(' / '.join(a))
	for i in range(1000):
		a = []
		for j in open("./Testing/{index}.txt".format(index=i), 'r', encoding='utf-8').read().strip().split(' / '):
			if j not in e and len(j) < 10: a.append(j)
		open("./Testing/{index}.txt".format(index=i), 'w', encoding='utf-8').write(' / '.join(a))