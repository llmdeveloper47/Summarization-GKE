import os
from torch import nn
import transformers
from transformers import AutoTokenizer
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


__version__ = "1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

max_input_length = 1024
max_target_length = 128

def download_artifacts():

    model_path = str(BASE_DIR) + "/model_artifacts"
    model_checkpoint ='t5-small'

    if os.path.isdir(model_path):


        print("Loading Model")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Model Loaded")
    else:
        print("No Model Artifacts Found")
        t5_model = None

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print("Tokenizer Loaded")
    
    return t5_model,tokenizer


t5_model,tokenizer = download_artifacts()


# input is of type list of string List[str], where each item in the list is an article which needs to be summarized
def predict_pipeline(text_input):

    text = ["summarize : " + item for item in text_input]

    inputs = tokenizer(text, max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt").input_ids

    outputs = t5_model.generate(inputs, max_length=max_target_length, do_sample=False, num_beams = 3)

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = [pred.strip() for pred in predictions]

    # result = pd.DataFrame(list(zip(text_input, predictions)))
    # result.columns = ['Text_Input','Summary']
    # result.to_csv("text_summary.csv")


    return predictions



# text_input = ["LONDON, England (Reuters) Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in Harry Potter and the Order of the Phoenix To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar, he told an Australian interviewer earlier this month. I don't think I'll be particularly extravagant. The things I like buying are things that cost about 10 pounds books and CDs and DVDs. At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film Hostel: Part II, currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. I'll definitely have some sort of party, he said in an interview. Hopefully none of you will be reading about it. Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. People are always looking to say kid star goes off the rails,he told reporters last month. But I try very hard not to go that way because it would be too easy for them. His latest outing as the boy wizard in Harry Potter and the Order of the Phoenix is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest.There is life beyond Potter, however. The Londoner has filmed a TV movie called My Boy Jack, about author Rudyard Kipling and his son, due for release later this year. He will also appear in December Boys, an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's Equus. Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: I just think I'm going to be more sort of fair game, he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."]

# curl -X 'POST' \
#   'https://5000-cs-330169620829-default.cs-us-east1-rtep.cloudshell.dev/predict' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "text": ["LONDON, England (Reuters) Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in Harry Potter and the Order of the Phoenix To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar, he told an Australian interviewer earlier this month. I don't think I'll be particularly extravagant. The things I like buying are things that cost about 10 pounds books and CDs and DVDs. At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film Hostel: Part II, currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. I'll definitely have some sort of party, he said in an interview. Hopefully none of you will be reading about it. Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. People are always looking to say kid star goes off the rails,he told reporters last month. But I try very hard not to go that way because it would be too easy for them. His latest outing as the boy wizard in Harry Potter and the Order of the Phoenix is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest.There is life beyond Potter, however. The Londoner has filmed a TV movie called My Boy Jack, about author Rudyard Kipling and his son, due for release later this year. He will also appear in December Boys, an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's Equus. Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: I just think I'm going to be more sort of fair game, he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."]
# }'


#text_input = ["Editor's note: In our Behind the Scenes series CNN correspondents share their experiences in covering news and analyze the stories behind the events. Here, Soledad O'Brien takes users inside a jail where many of the inmates are mentally ill. An inmate housed on the "forgotten floor," where many mentally ill inmates are housed in Miami before trial. MIAMI, Florida (CNN) -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the "forgotten floor." Here, inmates with the most severe mental illnesses are incarcerated until they're ready to appear in court. Most often, they face drug charges or charges of assaulting an officer --charges that Judge Steven Leifman says are usually "avoidable felonies." He says the arrests often result from confrontations with police. Mentally ill people often won't do what they're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid, delusional, and less likely to follow directions, according to Leifman. So, they end up on the ninth floor severely mentally disturbed, but not getting any real help because they're in jail. We toured the jail with Leifman. He is well known in Miami as an advocate for justice and the mentally ill. Even though we were not exactly welcomed with open arms by the guards, we were given permission to shoot videotape and tour the floor. Go inside the 'forgotten floor' » . At first, it's hard to determine where the people are. The prisoners are wearing sleeveless robes. Imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that's kind of what they look like. They're designed to keep the mentally ill patients from injuring themselves. That's also why they have no shoes, laces or mattresses. Leifman says about one-third of all people in Miami-Dade county jails are mentally ill. So, he says, the sheer volume is overwhelming the system, and the result is what we see on the ninth floor. Of course, it is a jail, so it's not supposed to be warm and comforting, but the lights glare, the cells are tiny and it's loud. We see two, sometimes three men -- sometimes in the robes, sometimes naked, lying or sitting in their cells. "I am the son of the president. You need to get me out of here!" one man shouts at me. He is absolutely serious, convinced that help is on the way -- if only he could reach the White House. Leifman tells me that these prisoner-patients will often circulate through the system, occasionally stabilizing in a mental hospital, only to return to jail to face their charges. It's brutally unjust, in his mind, and he has become a strong advocate for changing things in Miami. Over a meal later, we talk about how things got this way for mental patients. Leifman says 200 years ago people were considered "lunatics" and they were locked up in jails even if they had no charges against them. They were just considered unfit to be in society. Over the years, he says, there was some public outcry, and the mentally ill were moved out of jails and into hospitals. But Leifman says many of these mental hospitals were so horrible they were shut down. Where did the patients go? Nowhere. The streets. They became, in many cases, the homeless, he says. They never got treatment. Leifman says in 1955 there were more than half a million people in state mental hospitals, and today that number has been reduced 90 percent, and 40,000 to 50,000 people are in mental hospitals. The judge says he's working to change this. Starting in 2008, many inmates who would otherwise have been brought to the "forgotten floor" will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment, not just punishment. Leifman says it's not the complete answer, but it's a start. Leifman says the best part is that it's a win-win solution. The patients win, the families are relieved, and the state saves money by simply not cycling these prisoners through again and again. And, for Leifman, justice is served. E-mail to a friend ."]
# print(predict_pipeline(text_input))
