import json
from torch.utils.data import Dataset
import os
from math_verify import parse, verify
from verl.utils.reward_score.math_utils import remove_unit

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def filter_data(data: list[dict], filter_type: str = "default"):
    '''
    Filter out the problems with complex answer.
    
    Here, we assume simple answer is in the form of "<numeric value> <unit>"
    where <numeric value> is a number or a valid math expression and <unit> is a unit of measurement
    and they are separated by a single space.
    '''
    if filter_type == "default":
        filtered_answer = []
        for problem in data:
            answer_without_unit = remove_unit(problem['answer'])
            
            # deal with infinity
            answer_without_unit = answer_without_unit.replace("infinity", "\\infty")
            if "\\infty" not in answer_without_unit:
                answer_without_unit = answer_without_unit.replace("inf", "\\infty")
            answer_without_unit = answer_without_unit.replace("+\\inity", "\\infty")
            
            # parse the answer
            answer = parse(f"${answer_without_unit}$", extraction_mode="first_match")

            # if answer can be converted to a float, then it is a simple answer
            if len(answer) > 0 and is_number(answer[0]):
                problem['original_answer'] = problem['answer']
                problem['answer'] = str(float(answer[0]))
                filtered_answer.append(problem)
            # else:
            #     print(f"answer: {problem['answer']}, parsed_answer: {answer}")
            #     import ipdb; ipdb.set_trace()
    elif filter_type == "ipho":
        filtered_answer = []
        for problem in data:
            if "is_numerical" in problem and problem['is_numerical']:
                if "numerical_answer" in problem and is_number(problem['numerical_answer']):
                    problem["answer"] = str(problem['numerical_answer'])
                    filtered_answer.append(problem)
            elif "is_symbolic" in problem and problem['is_symbolic']:
                problem["answer"] = problem['latex_answer']
                # # parse the symbolic answer
                # parsed_answer = parse(f"${symbolic_answer}$")
                # print("--------------------------------")
                # print(f"symbolic_answer: {parsed_answer}")
                # print(f"gt             : {symbolic_answer}")
                # print("--------------------------------")
                filtered_answer.append(problem)
    elif filter_type == "hcv":
        filtered_answer = []
        for problem in data:
            if "answerType" in problem and problem['answerType'] == "equation" and "error" in problem and not problem['error']:
                problem["answer"] = problem['latex_answer']
                filtered_answer.append(problem)
            elif "answerType" in problem and problem['answerType'] == "numerical" and "numerical_answer" in problem and is_number(problem['numerical_answer']):
                problem['answer'] = str(problem['numerical_answer'])
                filtered_answer.append(problem)
    return filtered_answer


class HCVRealDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        chapters: list[int] = range(1, 13),   # all chapters
        use_simple_answer: bool = True
    ):
        """
        Args:
            dataset_dir: path to the dataset directory
            chapters: list of chapters to load
            use_simple_answer: if True, use simple answer only. Simple answer is the answer in the form of '<numeric value> <unit>'
        """
        self.dataset_dir = dataset_dir
        self.chapters = chapters

        self.data = []
        for chapter in self.chapters:
            try:
                with open(os.path.join(self.dataset_dir, f'Chapter_{chapter}/Chapter{chapter}_problems.json'), 'r') as f:
                    # self.data.extend(json.load(f))
                    data = json.load(f)
                    for problem in data:
                        problem['chapter'] = f'Chapter_{chapter}'
                        self.data.append(problem)
            except FileNotFoundError:
                print(f'Chapter {chapter} not found in dataset {self.__class__.__name__}')
        print(f'len(self.data): {len(self.data)}')
        if use_simple_answer:
            self.data = filter_data(self.data, filter_type="hcv")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class IPHORealDataset(Dataset):
    def __init__(
        self, 
        dataset_dir: str,
        chapters: list[str] = None,
        use_simple_answer: bool = True
    ):
        """
        Args:
            dataset_dir: path to the dataset directory
            chapters: list of chapters to load
            use_simple_answer: if True, use simple answer only. Simple answer is the answer in the form of '<numeric value> <unit>'
        """
        self.dataset_dir = dataset_dir
        self.chapters = chapters

        with open(os.path.join(self.dataset_dir, 'ipho_problems.json'), 'r') as f:
            self.data = json.load(f)

        if use_simple_answer:
            self.data = filter_data(self.data, filter_type="ipho")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    dataset = HCVRealDataset(
        dataset_dir='../../../../llm/hcv',
        chapters=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    print('len(dataset): ', len(dataset))
    print('dataset[0]: ', dataset[0])
    # # # save the dataset to a json file
    # # # DATA_DIR = os.getenv('DATA_DIR')
    # with open(f"/home/mprabhud/datasets/physics_sim_data/hcv_full/hcv_full_test.json", 'w') as f:
    #     json.dump(dataset.data, f)

    dataset = IPHORealDataset(
        dataset_dir='../../../../llm/ipho',
    )
    print('len(dataset): ', len(dataset))
    print('dataset[0]: ', dataset[0])

    # with open(f"/home/mprabhud/datasets/physics_sim_data/ipho_full/ipho_full_test.json", 'w') as f:
    #     json.dump(dataset.data, f)