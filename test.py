
n_actions = 3
m = 3

for i in range(n_actions**m):
	action_dict = {
		    j: ((i // (n_actions ** j)) % n_actions) for j in range(m)
		}

	print("[",end="")
	for key in action_dict.keys():
		print(f"{action_dict[key]} ",end="")
	print("]")
