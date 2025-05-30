#!/usr/bin/env python
# coding: utf-8

# In[3]:


import miditoolkit
import utils


# # Read MIDI (example)

# In[4]:


midi_obj = miditoolkit.midi.parser.MidiFile('./data/evaluation/000.midi')


# In[6]:


print(*midi_obj.instruments[0].notes, sep='\n')


# In[9]:


print(*midi_obj.tempo_changes[:10], sep='\n')


# # Convert to REMI events

# ## 1. Read midi into "Item"

# In[10]:


note_items, tempo_items = utils.read_items('./data/evaluation/000.midi')


# In[11]:


print(*note_items, sep='\n')


# In[13]:


print(*tempo_items[:10], sep='\n')


# ## 2. Quantize note items

# In[14]:


note_items = utils.quantize_items(note_items)


# In[15]:


print(*note_items, sep='\n')


# ## 3. extract chord (if needed)

# In[16]:


chord_items = utils.extract_chords(note_items)


# In[17]:


print(*chord_items, sep='\n')


# ## 4. group items

# In[18]:


items = chord_items + tempo_items + note_items
max_time = note_items[-1].end
groups = utils.group_items(items, max_time)


# In[19]:


for g in groups:
    print(*g, sep='\n')
    print()


# ## 5. "Item" to "Event"

# In[20]:


events = utils.item2event(groups)


# In[23]:


print(*events[:30], sep='\n')

