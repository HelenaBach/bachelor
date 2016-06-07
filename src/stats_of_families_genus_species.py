# devide dataset into families and count occurences of each family. 
import xml.etree.ElementTree as etree
import os
from shutil import copy
import json
import pickle 
relative_path = '../data/train/'


def create_and_store_pickle_dicts():
	families = {}
	genus = {}
	species = {}

	# loop through all data-files
	for file in os.listdir(relative_path):

		# only looking at the xml files
		if file.endswith('.xml'):
			tree = etree.parse(relative_path + file)
			# 
			root = tree.getroot()
#			id = file[:-4]
			id = root.find('MediaId').text
#			if not id is file[:4]:
#				print(id)
#				print(file[:4])
#
			Genus  = root.find('Genus').text
			family = root.find('Family').text
			specie = root.find('ClassId').text

			if Genus in genus:
				genus[Genus].append(id)
			else:
				genus[Genus] = [id]

			if family in families:
				families[family].append(id)
			else:
				families[family] = [id]

			if specie in species:
				species[specie].append(id)
			else:
				species[specie] = [id]

	stat_list = {'families' : families, 'genus' : genus, 'species' : species} 


	for name in stat_list:

		# save to file:
#		with open('../data/' + name + '.json', 'w') as f:
#		    json.dump(stat_list[name], f)
		with open( name + '.p', 'wb') as f:
			pickle.dump(stat_list[name], f)
	
		list_3  = {}
		list_5  = {}
		list_7  = {}
		list_10 = {}
		for key in stat_list[name]:
			number_of_occences = len(stat_list[name][key])
			if number_of_occences > 2:
				list_3[key] = stat_list[name][key]
			if number_of_occences > 4:
				list_5[key] = stat_list[name][key]
			if number_of_occences > 6:
				list_7[key] = stat_list[name][key]
			if number_of_occences > 9:
				list_10[key] = stat_list[name][key]
	
		# save to file:
		with open( name + '_>2.p', 'wb') as f:
		    pickle.dump(list_3, f)
	
		with open(name + '_>4.p', 'wb') as f:
		    pickle.dump(list_5, f)
		
		# save to file:
		with open( name + '_>6.p', 'wb') as f:
		    pickle.dump(list_7, f)
	
		# save to file:
		with open(name + '_>9.p', 'wb') as f:
		    pickle.dump(list_10, f)
	

	#load from file:
	'''
	with open('../data/genus.json', 'r') as f:
	    try:
	        data = json.load(f)
	    # if the file is empty the ValueError will be thrown
	    except ValueError:
	        data = {}
	'''

def save_number_of_occences(name, list):

	stats = {}

	for key in list:
		number_of_occences = len(list[key])
		stats[key] = number_of_occences

	# save to file:
	with open(name + '_stats.p', 'wb') as f:
	    pickle.dump(stats, f)


def print_the_list(name, list):

	real_number_3 = 0
	real_number_5  = 0
	real_number_7  = 0
	real_number_10 = 0
	for key in list: 
		if len(list[key]) > 2:
			real_number_3 += 1

		if len(list[key]) > 4:
			real_number_5 += 1

		if len(list[key]) > 6:
			real_number_7 += 1

		if len(list[key]) > 9:
			real_number_10 += 1

	
	print('number of ' + name + ': ' + str(len(list)))
	print('number of ' + name + ' > 2: ' + str(real_number_3))
	print('number of ' + name + ' > 4: ' + str(real_number_5))
	print('number of ' + name + ' > 6: ' + str(real_number_7))
	print('number of ' + name + ' > 9: ' + str(real_number_10))

def print_and_save_families_and_species(number):
	with open('../data/families.json', 'r') as f:
	    try:
	        families = json.load(f)
	    # if the file is empty the ValueError will be thrown
	    except ValueError:
	    	families = {}

	with open('../data/species_>' + str(number) + '.json', 'r') as f:
	    try:
	        species = json.load(f)
	    # if the file is empty the ValueError will be thrown
	    except ValueError:
	    	species = {}

	new_families = {}
	for key in families:
		for species__key in species:
			# this specie is a subset of the familie 
			if set(species[species__key]) < set(families[key]):
				if key in new_families:
					new_families[key] =  new_families[key] + species[species__key]
				else:
					new_families[key] = species[species__key]
	# save to file:
	with open('../data/families_and_species_<' + str(number) + '.json', 'w') as f:
	    json.dump(new_families, f)

	print('number of families with species > ' + str(number) + ': ' + str(len(new_families)))



create_and_store_pickle_dicts()

with open('families.p', 'rb') as f:
    try:
        families = pickle.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
    	families = {}

with open('species.p', 'rb') as f:
    try:
        species = pickle.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
    	species = {}

with open('genus.p', 'rb') as f:
    try:
        genus = pickle.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
    	genus = {}

stat_list = {'families' : families, 'genus' : genus, 'species' : species} 


for name in stat_list:
	print_the_list(name, stat_list[name])
	save_number_of_occences(name, stat_list[name])

print_and_save_families_and_species(2)
print_and_save_families_and_species(4)
print_and_save_families_and_species(6)
print_and_save_families_and_species(9)
print("{'LeafScan': 12605, 'Leaf': 13367}")

# JSON OBJECTS IS STORED IN THE DATA DIRECTORY


# Before the test set:
#number of species: 351
#number of species > 2: 223
#number of species > 4: 174
#number of species > 6: ls ..
#number of species > 9: 138
#number of genus: 241
#number of genus > 2: 162
#number of genus > 4: 128
#number of genus > 6: 102
#number of genus > 9: 93
#number of families: 87
#number of families > 2: 73
#number of families > 4: 69
#number of families > 6: 63
#number of families > 9: 56
#number of families with species > 2: 47
#number of families with species > 4: 40
#number of families with species > 6: 33
#number of families with species > 9: 31
#{'LeafScan': 12605, 'Leaf': 13367}


# After the test set: 
#number of species: 351
#number of species > 2: 223
#number of species > 4: 174
#number of species > 6: 150
#number of species > 9: 139
#number of families: 87
#number of families > 2: 73
#number of families > 4: 69
#number of families > 6: 63
#number of families > 9: 56
#number of genus: 241
#number of genus > 2: 162
#number of genus > 4: 128
#number of genus > 6: 103
#number of genus > 9: 94
#number of families with species > 2: 47
#number of families with species > 4: 40
#number of families with species > 6: 34
#number of families with species > 9: 32
#{'LeafScan': 12605, 'Leaf': 13367}
#