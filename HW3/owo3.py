import os

e = ['XDDDDD', 'span', 'gt', 'lt', 'QQ', 'XD', 'GG', 'gtgt', 'gtlt', 'ltlt', 'ltgt', 'ok', 'OK', 'Ok', 'xd', 'xD', 'XDD', 'QAQ', 'gtgtgt', 'ltltlt']
E = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job': 3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

if __name__ == '__main__':
	for i in E.keys():
		for j in os.listdir('./training/{key}_cut/'.format(key=i)):
			a = []
			for k in open("./training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().split(' / '):
				if k not in e and len(k) < 10: a.append(k)
			open("./training/{key}_cut/{file}".format(key=i, file=j), 'w', encoding='utf-8').write(' / '.join(a))

	for i in range(1000):
		a = []
		for j in open("./testing/{index}.txt".format(index=i), 'r', encoding='utf-8').read().strip().split(' / '):
			if j not in e and len(j) < 10: a.append(j)
		open("./testing/{index}.txt".format(index=i), 'w', encoding='utf-8').write(' / '.join(a))