from transformers import PreTrainedTokenizerFast
from typing import List
import random
import torch
random.seed(7777)

class ChatBot:
    """
    chatbot for temi.

    :arg QandA : Questions and Answer dataset.
                 The dataset consists of Question[str], Answer[str], Vector of Quetion[List[float]], place_id[int].
    """

    def __init__(self, QandA: List):
        self.QandA = QandA
        self.QandA_of_place = None
        self.pre_questions = None
        self.num_of_recommended_quetion = 10
        self.condition_of_similarity = 0.9
        self.place_range = list(range(-1, 14))
        self.unknown_answer = "죄송합니다. 질문을 잘 모르겠어요. 다시 질문해주세요."

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compress_tokenizer = PreTrainedTokenizerFast.from_pretrained("../tokenizer", use_cache=True)
        self.compress_model = torch.nn.Embedding(51200, 768).to(self.device)
        self.compress_model.load_state_dict(torch.load("./embedding_model_state.pt"))

    def question(self, user_question: str) -> str:
        """
        Ask a Question to Chatbot.\n
        return the most similar answer in QnA database.\n
        have to process setPlace method before.\n
        if the answer does NOT exist,\n
        return the prepared answer and reflect this question in next recommended questions.\n

        Example
        -------
        >>> user_text = "여기는 어디인가요?"
        >>> QnA_database = [["여기가 어딘가요?", "여기는 학교의 정문입니다.", torch.rand(768, dtype=torch.float), 0]]
        >>>
        >>> chatbot = ChatBot(QnA_database)
        >>> chatbot.setPlace(0)
        >>> re_ques = chatbot.make_recommended_questions()
        >>> bot_answer = chatbot.question(user_text)

        :param user_question: user's question.
        :return: answer about given question.
        """
        assert user_question and self.QandA_of_place
        user_question = self._encoding_question(user_question)

        best_similarity = self.condition_of_similarity
        answer = self.unknown_answer

        for _, a, question_vector in self.QandA_of_place:
            turn_similarity = self._get_cosine_similarity(question_vector, user_question)
            if turn_similarity > best_similarity:
                best_similarity = turn_similarity
                answer = a
        if answer == self.unknown_answer:
            self.pre_questions = user_question
        return answer

    def make_recommended_questions(self) -> List[str]:
        """
        generate recommended questions.\n
        the number of recommended questions are defined in beforehand.\n
        have to process setPlace method before.\n
        if the question numbers are less than recommended question numbers,\n
        just return all of questions.\n

        Example
        -------
        >>> user_text = "여기는 어디인가요?"
        >>> QnA_database = [["여기가 어딘가요?", "여기는 학교의 정문입니다.", torch.rand(768, dtype=torch.float), 0]]
        >>>
        >>> chatbot = ChatBot(QnA_database)
        >>> chatbot.setPlace(0)
        >>> re_ques = chatbot.make_recommended_questions()
        >>> bot_answer = chatbot.question(user_text)

        :return: generated recommended qustions
        """
        assert self.QandA_of_place

        questions = self.QandA_of_place
        if self.pre_questions is not None:
            sim_quesions = [[self._get_cosine_similarity(self.pre_questions, v), q] for q, _, v in questions]
            sorted_quetions = sorted(sim_quesions, reverse=True)
            questions = [q for _, q in sorted_quetions]
            self.pre_questions = None
        else:
            random.shuffle(questions)
        return questions[:self.num_of_recommended_quetion]

    def setPlace(self, place_id: int):
        """
        setting place of questions.\n
        maybe can be explained that changing range of searching.\n
        if you try to input doesn't exist place, it cause error.\n

        Example
        -------
        >>> user_text = "여기는 어디인가요?"
        >>> QnA_database = [["여기가 어딘가요?", "여기는 학교의 정문입니다.", torch.rand(768, dtype=torch.float), 0]]
        >>>
        >>> chatbot = ChatBot(QnA_database)
        >>> chatbot.setPlace(0)
        >>> re_ques = chatbot.make_recommended_questions()
        >>> bot_answer = chatbot.question(user_text)

        :param place_id: the id of place. The database has to have questions of this place.
        :return: None
        """
        assert place_id in self.place_range, "place id has to include in place range"
        self.QandA_of_place = filter(lambda x: x[3] in [place_id, -1], self.QandA)

    def _get_cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        v1 = torch.FloatTensor(v1).unsqueeze(0)
        v2 = torch.FloatTensor(v2).unsqueeze(0)
        return torch.cosine_similarity(v1, v2).item()

    def _encoding_question(self, ques: str) -> torch.FloatTensor:
        encoded_ques = self.compress_tokenizer.encode(ques, return_tensors="pt")  # (1, token_num)
        embedding_vector = self.compress_model(encoded_ques)  # (1, token_num, embedding_dim(768))
        embedding_vector = torch.mean(embedding_vector, dim=1).squeeze()  # (embedding_dim)
        return embedding_vector
