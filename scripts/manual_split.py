import cPickle as pickle

original_label_fname = 'fine_label_names.txt'
manual_label_fname = 'fine_label_names_mansplit.txt'
output_fname = 'manual_split.pkl'


# Load CIFAR-100 labels
with open(original_label_fname) as fd:
    label_original = [t.strip() for t in fd.readlines()]
with open(manual_label_fname) as fd:
    label_manual = [t.strip() for t in fd.readlines()]

# Match original and manual labels
class_map = []
for l in label_manual:
    class_map.append(label_original.index(l))

# Manual split - [[6, 4, 3], [3, 4]]
print('Manual split\n')
output = [[class_map[:30], class_map[30:50], class_map[50:65]], [class_map[65:80],class_map[80:]]]
for i1, c1 in enumerate(output):
    for i2, ci2 in enumerate(c1):
        print 'Cluster %d-%d: ' % (i1+1, i2+1) ,
        for idx in ci2:
            print label_original[idx] ,
        print ' '
print ' '

# Save as pkl file
print('Save as pkl file')
with open(output_fname, 'wb') as fd:
    pickle.dump(output, fd)

print('Done!')
