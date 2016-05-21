import random
import json
import os
import sys

# shuffles and partition a list into three sublists.
def shuffle_and_partition(list):
	# shuffle the list
	random.shuffle(list)

	# partition the list in 5 equally sized lists
	return [list[i::5] for i in range(5)]


def split_training_test(number, path='../data/species_>'):

	with open( path + str(number) + '.json', 'r') as f:
		try:
			species = json.load(f)
    	# if the file is empty the ValueError will be thrown
		except ValueError:
			species = {}

	division = {'test' : [], 'train' : []}

	for key in species:
		sublists = shuffle_and_partition(species[key])
		# test is the first list -> if < 5 and if 5 is not divisible
		# test is between 20% and 33,3% of the whole set
		test = sublists.pop(0)
		# merge all other sublists
		train = [item for sublist in sublists for item in sublist]
		division['test']  = division['test'] + test
		division['train'] = division['train'] + train

	print('number of items in train ( species number >' + str(number) + '): ' + str(len(division['train'])))
	print('number of items in test ( species number >' + str(number) + '): ' + str(len(division['test'])))
    # save to file:
	with open('../data/data_division.json', 'w') as f:
	    json.dump(division, f)


# destination path should be created on beforehand
def move_file(current_path, destination_path, file_name):
	os.rename(current_path + file_name, destination_path + file_name)

def move_all_files(current_path, destination_path_test, destination_path_train, division_path):
	with open( division_path, 'r') as f:
		try:
			division = json.load(f)
    	# if the file is empty the ValueError will be thrown
		except ValueError:
			division = {}

	print('number of duplicates in test set: ' + str(len(list_duplicates(division['test']))))
	if len(list_duplicates(division['test'])) > 0:
		print(list_duplicates(division['test']))

	for id in set(division['test']):
		move_file(current_path, destination_path_test, id + '.xml')
		move_file(current_path, destination_path_test, id + '.jpg')

	print('number of duplicates in train set: ' + str(len(list_duplicates(division['train']))))
	if len(list_duplicates(division['train'])) > 0:
		print(list_duplicates(division['train']))

	for id in set(division['train']):
		move_file(current_path, destination_path_train, id + '.xml')
		move_file(current_path, destination_path_train, id + '.jpg')


def list_duplicates(seq):
  seen = set()
  seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
  seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
  return list( seen_twice )

#def rename_file():
#	# loop through all data-files
#	for file in os.listdir('../data/train/'):
#		if file[:8] == 'leafscan':
#			os.rename('../data/train/' + file, '../data/train/' + file[8:])

#split_training_test(2)
print('number of files in leafscan :' +  str(len(os.listdir('../../Data/leafscan/'))))
print('number of files in train    :' +  str(len(os.listdir('../../Data/train/'))))
print('number of files in test     :' +  str(len(os.listdir('../../Data/test/'))))
move_all_files('../../Data/leafscan/', '../../Data/test/', '../../Data/train/', 'data_division.json')
print('after split:')
print('number of files in leafscan :' +  str(len(os.listdir('../../Data/leafscan/'))))
print('number of files in train    :' +  str(len(os.listdir('../../Data/train/'))))
print('number of files in test     :' +  str(len(os.listdir('../../Data/test/'))))

#number of items in train ( species number >2): 10037
#number of items in test ( species number >2): 2613
# all = 12650 * 2 = 25300

#number of files in leafscan :25652
#number of files in train    :0
#number of files in test     :0
#number of duplicates in test set: 0
#number of duplicates in train set: 0
#after split:
#number of files in leafscan :352
#number of files in train    :20074
#number of files in test     :5226
# 20074 + 5226 = 25300 hurra! :D
