import numpy as np
import math
from sklearn.neural_network import MLPClassifier

def derive(L_sys, n):
	nextWord = ''
	currWord = L_sys['w']
	for i in range(n):
		for char in currWord:
			if char in L_sys['P']:
				nextWord += L_sys['P'][char]
			else:
				nextWord += char
		currWord = nextWord
	
	return currWord

def expertData(q_init,d,delta,word):

	X = [q_init]
	labels = [c for c in word]
	curr_q = []
	for c in word:
		if c=='F':
			curr_q = [q_init[0]+d*np.cos(q_init[2]),
					  q_init[1]+d*np.sin(q_init[2]),
					  q_init[2],q_init[3]+1]
		elif c=='+':
			alpha = q_init[2]+delta
			if alpha > 2*np.pi:
				alpha-=2*np.pi
			curr_q = [q_init[0], q_init[1], alpha,q_init[3]+1]
		elif c=='-':
			alpha = curr_q[2]-delta
			if alpha < 0:
				alpha+=2*np.pi
			curr_q = [q_init[0], q_init[1], alpha,q_init[3]+1]
		
		X.append(curr_q)

	return X,labels

def step(q,a,id_to_label,delta,d):
	curr_q = q

	if id_to_label[a] == 'F':
		curr_q = [curr_q[0]+d*np.cos(curr_q[2]),
				  curr_q[1]+d*np.sin(curr_q[2]),
				  curr_q[2],curr_q[3]+1]
	if id_to_label[a] == 'f':
		curr_q = [curr_q[0]+d*np.cos(curr_q[2]),
				  curr_q[1]+d*np.sin(curr_q[2]),
				  curr_q[2],curr_q[3]+1]
	elif id_to_label[a] == '+':
		alpha = curr_q[2]+delta
		if alpha > 2*np.pi:
			alpha-=2*np.pi
		curr_q = [curr_q[0], curr_q[1], alpha,curr_q[3]+1]
	elif id_to_label[a] == '-':
		alpha = curr_q[2]-delta
		if alpha < 0:
			alpha+=2*np.pi
		curr_q = [curr_q[0], curr_q[1], alpha,curr_q[3]+1]

	return curr_q

def correctiveManouver(start, end, label_to_id):
	manouver = ''

	v = np.array(end) - np.array(start)
	x_comp_angle = math.atan2(0, v[0])
	y_comp_angle = math.atan2(v[1], 0)


	if x_comp_angle < 0:
		x_comp_angle += 2*np.pi

	if y_comp_angle < 0:
		y_comp_angle += 2*np.pi


	if v[0] != 0:
		sym=''
		if x_comp_angle-start[2] < 0:
			# turn right -
			sym = '-'
		elif x_comp_angle-start[2] > 0:
	    	# turn left
			sym = '+'
	    
		n_turns = np.abs(x_comp_angle-start[2])/(np.pi/2)
		manouver += sym*np.int(n_turns)
		#print(manouver)
		manouver+='f'*np.abs(np.int(v[0]))
		#print(manouver)
	else:
		x_comp_angle = start[2]


	if v[1] != 0:
		sym=''
		if y_comp_angle-x_comp_angle < 0:
	    	# turn right -
			sym = '-'
		elif y_comp_angle-x_comp_angle > 0:
	    	# turn left
			sym = '+'

		n_turns = np.abs(y_comp_angle-x_comp_angle)/(np.pi/2)
		manouver += sym*np.int(n_turns)
		#print(manouver)

		manouver+='f'*np.abs(np.int(v[1]))
		#print(manouver)
	else:
		y_comp_angle = x_comp_angle



	if end[2]-y_comp_angle < 0:
    	# turn right -
		sym = '-'
	elif end[2]-y_comp_angle > 0:
    	# turn left
		sym = '+'

	n_turns = np.abs(end[2]-y_comp_angle)/(np.pi/2)
	manouver += sym*np.int(n_turns)

	#print(manouver)


	return manouver, [label_to_id[c] for c in manouver]

def acc(x,y):
	a = [int(i==j) for i,j in zip(x,y)];
	return sum(a)/len(a);








G = {
	'V':['F','+','-'],
	'w':'F+F+F+F',
	'P':{'F':'FF+F+F+F+F+F-F'}
}


word = derive(G,1)

print(word)

q_0 = [0,0,0,0]
label_to_id = {'F':0, '+':1, '-':2, 'f':3}
id_to_label = {0:'F', 1:'+', 2:'-', 3:'f'}


X, labels = expertData(q_0, 1, np.pi/2, word)
X.pop()
y = [label_to_id[l] for l in labels]


acc_X, acc_y = [],[]

epsilon = 0.9
for i in range(500):
	q = q_0[:]
	X_pi, y_pi = [], []
	X_pi_corrected, y_pi_corrected = [],[]

	if np.random.uniform() <= -(0.4/500)*i+0.5:
		print('EXPERT')
		X_pi_corrected, y_pi_corrected = X[:],y[:]
	else:
		print('LEARNER')
		for t in range(len(X)):
			a = pi.predict([q])[0]
			q_new  = step(q,a,id_to_label,np.pi/2,1)
			
			X_pi.append(q)
			y_pi.append(a)
			
			q = q_new[:]

		
		m_counter,m_length = 0,0
		print(acc(y,y_pi))
		for t in range(len(X)-2):
			if y[t]!=y_pi[t] :
				
				q_pi_wrong = X_pi[t+1][:]
				q_e = X[t+2][:]



				if np.linalg.norm(np.array(q_pi_wrong)-np.array(q_e))>4:
					continue
				else:
					m_counter+=1
				manouver,manouverIds = correctiveManouver(q_pi_wrong,q_e,label_to_id)
				m_length+=len(manouver)

				#print(manouver)

				#print([(round(i)) for i in q_pi_wrong], [round(i) for i in q_e], manouver)

				X_pi_corrected.append(q_pi_wrong)

				for ids in manouverIds:
					y_pi_corrected.append(ids)
					q_new_corrected = step(q_pi_wrong, ids ,id_to_label, np.pi/2,1)
					X_pi_corrected.append(q_new_corrected)
					q_pi_wrong = q_new_corrected
				
				X_pi_corrected.pop()

				#print(m_counter, m_length/m_counter)

	acc_X, acc_y = acc_X+X_pi_corrected, acc_y+y_pi_corrected
	print(len(acc_X),len(acc_y))
	


	pi = MLPClassifier(random_state=1, max_iter=600, hidden_layer_sizes=(20,)).fit(acc_X, acc_y)


	word = ''
	q = q_0[:]
	for t in range(300):
		a = pi.predict([q])[0]
		q_new  = step(q,a,id_to_label,np.pi/2,1)
		
		#X_pi.append(q)
		word+=id_to_label[a]
		
		q = q_new[:]

	print(word)

	



# from pprint import pprint

# for i,j,k,l,m,n in zip(X[:20],
# 	y[:20],
# 	X_pi[:20],
# 	y_pi[:20],
# 	X_pi_corrected[:20],
# 	y_pi_corrected[:20]):
# 	print([round(e) for e in i],j,[round(e) for e in k],l,[round(e) for e in m],n)




