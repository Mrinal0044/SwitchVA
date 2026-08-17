import os
import re
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# The expanded, verified list of aspect keywords from the dataset
ASPECT_KEYWORDS = [
    'local train', 'pollution', 'maid', 'garmi', 'cab', 'commute', 'road', 'gym',
    'song', 'standup comedy', 'podcast', 'interview', 'trailer', 'web series', 'show',
    'charger', 'trimmer', 'kurti', 'jacket', 'earphones', 'camera', 'powerbank', 'smartwatch', 'washing machine',
    'client call', 'promotion', 'wfh', 'deadline', 'salary hike',
    'ganesh chaturthi', 'eid', 'festival', 'navratri', 'diwali', 'makar sankranti', 'holi',
    'tournament', 'ipl', 'cricket', 'test series', 'world cup', 'team',
    'cafe', 'pizza', 'dessert', 'momo', 'ubrger', 'thali', 'biryani',
    'dost', 'boyfirend', 'date', 'breakup', 'crush', 'family trip', 'best friend',
    'online class', 'degree', 'tuition', 'canteen', 'viva', 'syllabus', 'ersult', 'result', 'professor',
    'chunav', 'tax', 'protest', 'scheme', 'government', 'election', 'policy', 'rally', 'news',
    'laptop', 'ac', 'tv', 'headphone', 'tablet', 'phone', 'shoes', 'fridge', 'bag',
    'gst', 'budget', 'neta', 'govenrment',
    'dosa', 'dhaba', 'samosa', 'burger', 'street food', 'chai', 'pani puri',
    'vlog', 'film', 'music video', 'movie', 'web esries',
    'hostel', 'college', 'assignment', 'exam', 'tuitoin',
    'football', 'badminton', 'fotoball', 'crciket', 'match', 'olympics',
    'onam', 'mela', 'durga puja', 'christmas',
    'wifi', 'metro', 'bijli ka bill', 'monsoon', 'grocery', 'paani', 'traffic', 'baarish',
    'boss', 'appraisal', 'work from home', 'shift', 'hr', 'maanger', 'project',
    'shaadi', 'ex', 'rishtedaar', 'girlfriend', 'boyfriend',
    'interivew', 'niterview', 'dhbaa', 'scooter', 'phoen', 'headphones', 'shoes', 'fridge', 'bag', 'laptop', 'chilla-fy'
]

# Sort keywords by length descending to match the longest phrase first
ASPECT_KEYWORDS = sorted(list(set(ASPECT_KEYWORDS)), key=len, reverse=True)


def find_aspect_word_indices(words_list, clean_text):
    """
    Finds the start and end word indices of the aspect term in the words_list.
    """
    text_lower = " ".join(words_list).lower()
    
    # Find the matching keyword
    matched_kw = None
    matched_idx = -1
    
    for kw in ASPECT_KEYWORDS:
        # Check if keyword is in the joined text
        # We look for word boundary matches to avoid partial word matches
        pattern = r'\b' + re.escape(kw) + r'\b'
        match = re.search(pattern, text_lower)
        if match:
            matched_kw = kw
            matched_idx = match.start()
            break
            
    if matched_kw is None:
        # Fallback: substring matching
        for kw in ASPECT_KEYWORDS:
            if kw in text_lower:
                matched_kw = kw
                matched_idx = text_lower.find(kw)
                break
                
    if matched_kw is None:
        # Default to the first word if nothing matches
        return 0, 1
        
    # Determine which words correspond to the character span [matched_idx, matched_idx + len(matched_kw)]
    char_start = matched_idx
    char_end = matched_idx + len(matched_kw)
    
    word_start = -1
    word_end = -1
    
    curr_char = 0
    for idx, w in enumerate(words_list):
        word_len = len(w)
        word_char_start = curr_char
        word_char_end = curr_char + word_len
        
        # Check overlap
        if word_char_end > char_start and word_start == -1:
            word_start = idx
        if word_char_end >= char_end and word_end == -1:
            word_end = idx + 1
            break
            
        curr_char += word_len + 1 # +1 for space
        
    if word_start == -1:
        word_start = 0
    if word_end == -1:
        word_end = word_start + 1
        
    return word_start, word_end


class SwitchVADataset(Dataset):
    def __init__(self, csv_path, word_languages_path, split=None, tokenizer_name='xlm-roberta-base', max_len=128):
        df_all = pd.read_csv(csv_path)
        if split is not None:
            self.df = df_all[df_all['split'] == split].reset_index(drop=True)
        else:
            self.df = df_all
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load word language lookup
        with open(word_languages_path, 'r') as f:
            self.word_languages = json.load(f)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['clean_text'])
        words_list = text.split()
        
        # 1. Assign word-level languages
        word_langs = []
        for w in words_list:
            w_clean = re.sub(r'[^a-z0-9\-]', '', w.lower())
            lang = self.word_languages.get(w_clean, 'hindi')
            if lang == 'hindi':
                word_langs.append(0)
            elif lang == 'english':
                word_langs.append(1)
            else:
                word_langs.append(2)
                
        # 2. Extract aspect word indices
        word_start, word_end = find_aspect_word_indices(words_list, text)
        
        # 3. Tokenize and align
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)
        
        # Align word-level languages and aspect span to subword tokens
        token_langs = []
        aspect_token_mask = torch.zeros(self.max_len, dtype=torch.float32)
        
        aspect_token_indices = []
        
        for t_idx, w_id in enumerate(word_ids):
            if w_id is None:
                # Special token
                token_langs.append(2) # Other
            else:
                if w_id < len(word_langs):
                    token_langs.append(word_langs[w_id])
                else:
                    token_langs.append(2)
                    
                if word_start <= w_id < word_end:
                    aspect_token_indices.append(t_idx)
                    aspect_token_mask[t_idx] = 1.0
                    
        # Pad token languages to max_len
        while len(token_langs) < self.max_len:
            token_langs.append(2)
        token_langs = token_langs[:self.max_len]
        token_langs = torch.tensor(token_langs, dtype=torch.long)
        
        # 4. Compute switch boundaries
        # switch_boundary[t] = 1 if language changes between token t and t+1, else 0
        switch_boundary = torch.zeros(self.max_len, dtype=torch.long)
        for t in range(self.max_len - 1):
            # Only consider switches between Hindi (0) and English (1)
            l1 = token_langs[t].item()
            l2 = token_langs[t+1].item()
            if (l1 == 0 and l2 == 1) or (l1 == 1 and l2 == 0):
                switch_boundary[t] = 1
                
        # Find indices of switch boundaries
        switch_indices = (switch_boundary == 1).nonzero(as_tuple=True)[0].tolist()
        
        # 5. Compute SPE features for the aspect span
        if len(aspect_token_indices) > 0:
            asp_start = min(aspect_token_indices)
            asp_end = max(aspect_token_indices) + 1
        else:
            asp_start = 0
            asp_end = 1
            
        # Distance, Density, Direction
        # default values when there are no switches
        min_dist = 999.0
        nearest_switch_idx = -1
        
        for s_idx in switch_indices:
            # Distance computation:
            if s_idx < asp_start:
                dist = float(asp_start - s_idx - 1)
            elif s_idx >= asp_end:
                dist = float(s_idx - asp_end + 1)
            else:
                dist = 0.0 # Switch falls inside the aspect
                
            if dist < min_dist:
                min_dist = dist
                nearest_switch_idx = s_idx
                
        # Density: number of switch boundaries within window of size W=5 around aspect
        window_start = max(0, asp_start - 5)
        window_end = min(self.max_len, asp_end + 5)
        density = float(sum(1 for s_idx in switch_indices if window_start <= s_idx < window_end))
        
        # Direction: 0 = No switch, 1 = Hindi -> English, 2 = English -> Hindi
        direction = 0
        if nearest_switch_idx != -1:
            l_curr = token_langs[nearest_switch_idx].item()
            l_next = token_langs[nearest_switch_idx + 1].item()
            if l_curr == 0 and l_next == 1:
                direction = 1 # Hindi -> English
            elif l_curr == 1 and l_next == 0:
                direction = 2 # English -> Hindi
                
        # Format SPE features vector: [proximity, density, direction_is_hi_to_en, direction_is_en_to_hi]
        # We transform distance to proximity = 1.0 / (1.0 + min_dist) if switches exist, else 0.0
        # We also scale density by 5.0 to keep it in a small range
        proximity = 1.0 / (1.0 + min_dist) if nearest_switch_idx != -1 else 0.0
        spe_vector = torch.tensor([
            proximity,
            density / 5.0,
            1.0 if direction == 1 else 0.0,
            1.0 if direction == 2 else 0.0
        ], dtype=torch.float32)
        
        valence = float(row['computed_valence'])
        arousal = float(row['computed_arousal'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_langs': token_langs,
            'aspect_token_mask': aspect_token_mask,
            'spe_features': spe_vector,
            'min_dist': min_dist,  # Actual distance (999.0 if none)
            'valence': torch.tensor(valence, dtype=torch.float32),
            'arousal': torch.tensor(arousal, dtype=torch.float32),
            'domain': str(row['domain'])
        }


if __name__ == "__main__":
    # Quick sanity check
    dataset = SwitchVADataset(
        csv_path='/Users/kmrinal/SwitchVA/dataset.csv',
        word_languages_path='/Users/kmrinal/SwitchVA/word_languages.json',
        tokenizer_name='xlm-roberta-base'
    )
    print(f"Loaded dataset of size: {len(dataset)}")
    
    # Print a sample's processed details
    sample = dataset[0]
    print("\nSample 0:")
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Token languages:", sample['token_langs'][:20])
    print("Aspect token mask active indices:", sample['aspect_token_mask'].nonzero().squeeze(1).tolist())
    print("SPE Features (dist, dens, hi_to_en, en_to_hi):", sample['spe_features'].tolist())
    print("Valence:", sample['valence'].item(), "Arousal:", sample['arousal'].item())
