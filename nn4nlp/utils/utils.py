import re
import spacy
import torch
import logging
import wikipediaapi

from tqdm.auto import tqdm
from pprint import PrettyPrinter
from sklearn.metrics import accuracy_score, f1_score
from typing import (
    List, Callable, Optional, Dict, Tuple
)

# Configure spacy
nlp = spacy.load("es_core_news_sm")

# Configure wiki api
user_agent = "my-wikipedia-client/1.0 (https://yourwebsite.com/contact)"
lang = 'es'
wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)

# Configure pretty printer
pp = PrettyPrinter(compact=True)
print_ = lambda x: pp.pprint(x)


class CommonTools:
    '''
    Dummy class to aggregate helper functions
    '''

    @staticmethod
    def eval_response_with_ideal(
                prompt_template: Callable[[str], str],
                responses_with_ideals: List[Dict[str,List]],
                get_completion: Callable[[str], str],
                debug: Optional[bool]=True
            ) -> float:
        if debug:
            print('')
            print('='*30)
            print("---- Prompt template used: ----")
            print(f"{prompt_template('')}")
            print('='*30)
        total_correct = 0
        i = 0
        for pair in tqdm(responses_with_ideals, desc='Evaluating model responses'):
            i += 1
            student_input = pair["student_input"]
            correct_answer = pair["correct_answer"]
            prompt = prompt_template(student_input)
            response = get_completion(prompt)
            response = response.lower().strip() # make lowercase and eliminate spaces
            correct = 1 if response in correct_answer else 0
            total_correct += correct
            if debug:
                print('')
                print('-'*30)
                print(f"Student's input {i}:")
                print(f"\t{student_input}")
                print(f"Model's response {i}:")
                print(f"\t{response}")       
                if correct == 1:
                    print(f"Response correct!")
                else:
                    print(f"Incorrect!")
                    print(f"Correct response should be: {correct_answer}")
        correctness = total_correct / len(responses_with_ideals) * 100
        if debug:
            print('')
            print(f"Percentage of correct responses: {correctness}")
        return correctness


    @staticmethod
    def clean_text(text:str) -> str:
        """
        Cleans the provided text by removing citation markers (e.g., [24])
        and unwanted characters like the zero-width space (\u200b).

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Remove anything inside square brackets, e.g. [24] or [citation needed]
        cleaned_text = re.sub("\\[\\d+\\]\u200b", '', text)
        
        # Remove symbols
        cleaned_text = re.sub(re.escape('[\p{P}]'), '', cleaned_text) 

        # Remove extra whitespace created after removals
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()


        
        return cleaned_text


    @staticmethod
    def get_wikis_from_titles(title_list: List[str]) -> List[List[str]]:
        """
        Retrieve and process Wikipedia pages based on a list of page titles.

        For each title in `title_list`, this function:
        1. Retrieves the corresponding Wikipedia page using the `wiki` module.
        2. Cleans the page text using the `clean_text` function.
        3. Processes the cleaned text with an NLP model (`nlp`) to extract sentences.
        4. Aggregates the extracted sentences into a list.

        Parameters:
            title_list (List[str]): A list of Wikipedia page titles.

        Returns:
            List[List[str]]: A list where each element is a list of sentences extracted
                            from the corresponding Wikipedia page.

        Notes:
            - It is assumed that `wiki`, `clean_text`, and `nlp` are available in the global scope.
            - If a page cannot be retrieved (e.g., due to a disambiguation error or page not found),
            the function logs the error and skips that title.
        """
        # Initialize an empty list to store the sentence lists for each Wikipedia page.
        all_sentences: List[List[str]] = []

        # Iterate over each title in the provided list.
        for title in title_list:
            try:
                # Attempt to retrieve the Wikipedia page for the current title.
                page = wiki.page(title)
            except Exception as e:
                # Log the error with the problematic title and skip to the next title.
                logging.error(f"Failed to retrieve page for title '{title}': {e}")
                continue

            # Clean the raw text from the Wikipedia page.
            cleaned_text = NN4NLPUtils.clean_text(page.text)

            # Process the cleaned text with the NLP model to obtain a document object.
            doc = nlp(cleaned_text)

            # Extract sentences from the document using the NLP model's sentence segmentation.
            for sentence in doc.sents:
                # Append the sentence for this page to the overall collection.
                all_sentences.append(sentence.text)

        # Return the collection of sentence lists.
        return all_sentences
    
    @staticmethod
    def decode_token_list(encoded_str):
        """
        Decodes a string representing a list of tokens.
        Tokens are expected to be enclosed in either single or double quotes.
        The function handles tokens that are commas as well as tokens that
        include escaped characters.

        For example:
        coded_list = r"['[CLS]', '[MASK]', 'is', 'a', ',', 'there', \"'s\", 'little']"
        returns:
        ['[CLS]', '[MASK]', 'is', 'a', ',', 'there', "'s", 'little']

        Args:
            encoded_str (str): The encoded list as a string.
        
        Returns:
            list: A list of decoded tokens.
        """
        s = encoded_str.strip()
        # Remove the outer square brackets if present.
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        
        tokens = []
        i = 0
        n = len(s)
        
        while i < n:
            # Skip any whitespace or commas between tokens.
            while i < n and s[i] in " \t\n\r,":
                i += 1
            if i >= n:
                break
            
            # If a token starts with a quote, parse the quoted token.
            if s[i] in ("'", '"'):
                quote_char = s[i]
                i += 1  # Skip the opening quote.
                token_chars = []
                while i < n:
                    # Handle escape sequences.
                    if s[i] == '\\' and i + 1 < n:
                        token_chars.append(s[i+1])
                        i += 2
                    # End of token when we encounter the matching closing quote.
                    elif s[i] == quote_char:
                        i += 1  # Skip the closing quote.
                        break
                    else:
                        token_chars.append(s[i])
                        i += 1
                tokens.append(''.join(token_chars))
            else:
                # Fallback: if token is not quoted, accumulate until the next comma.
                token_chars = []
                while i < n and s[i] != ',':
                    token_chars.append(s[i])
                    i += 1
                tokens.append(''.join(token_chars).strip())
        
        return tokens

    @staticmethod
    def evaluate_classification(dataloader, model, criterion) -> Tuple[float, float, float]:
        '''Evaluate a classification model'''

        model.eval()  # Turn off dropout and other training-specific behaviors

        total_loss = 0
        total_batches = 0
        total_count = 0
        y_true, y_predict = list(), list()
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            for inputs_batch, classes_batch in dataloader:
                # Forward pass
                prediction = model(inputs_batch)

                # Calculate loss 
                loss = criterion(prediction, classes_batch)

                total_loss += loss.item()
                total_batches += 1

                logits = torch.softmax(prediction, dim=1)
                prediction = torch.argmax(logits, dim=1)         
                total_count += classes_batch.size(0)
                y_true.extend(classes_batch.view(-1).cpu().numpy().tolist())
                y_predict.extend(prediction.cpu().numpy().tolist())

        avg_loss = total_loss / (total_batches + 1)
        #print(f"Average Loss: {avg_loss:.4f}, Average Next Sentence Loss: {avg_next_sentence_loss:.4f}, Average Mask Loss: {avg_mask_loss:.4f}")
        acc = accuracy_score(y_true, y_predict)
        f1 = f1_score(y_true, y_predict)    
        #print(f"Accuracy: {acc}")
        #print(f"F1 score: {f1}")
        return avg_loss, acc, f1