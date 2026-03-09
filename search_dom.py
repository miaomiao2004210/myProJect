with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()
    
idx = content.find('DOMContentLoaded')
print(f'Found at index: {idx}')

if idx >= 0:
    print(f'Context: {content[max(0,idx-100):idx+200]}')
else:
    print('NOT FOUND')
    
# Also search for the upload button
idx2 = content.find('openFileInput')
print(f'\nopenFileInput found at: {idx2}')
if idx2 >= 0:
    print(f'Context: {content[max(0,idx2-50):idx2+100]}')
