from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageDraw
import torch
import os

class SimpleInference:
    def __init__(self, model_name_or_path: str):
        """
        Args:
            model_name_or_path (str): The name or path of the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).eval().cuda()
        self.model.tokenizer = self.tokenizer

    def inference(self,
                  prompt: str,
                  image_path: str,
                  task: str,
                  plot: bool=False,
                  save_path: str=None,
                  enable_thinking: bool=False,
                  do_sample: bool=True):
        """
        A simple inference function.
        """
        pred = self.simple_chat(prompt, image_path, enable_thinking=enable_thinking, do_sample=do_sample)

        if task == "general":
            if plot:
                print("General task does not support plotting.")
            return pred
        elif task in ["grounding", "affordance", "trajectory", "pointing"]:
            if plot:
                if not os.path.exists(image_path):
                    print(f"Image path '{image_path}' does not exist for plotting.")
                    return pred
                
                self.plot_results(image_path, pred["answer"], task, save_path)
            return pred
        else:
            raise ValueError(f"Task '{task}' is not supported.")

    def plot_results(self, image_path, answer, task, save_path=None):
        """
        Plotting results on the image.
        """
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            if task == "grounding" or task == "affordance":
                xy = eval(answer)
                draw.rectangle(xy, outline="red", width=5)
            elif task == "trajectory":
                points = eval(answer)
                if len(points) > 1:
                    draw.line(points, fill="red", width=5)
                for point in points:
                    draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill="green")
            elif task == "pointing":
                points = eval(answer)
                for point in points:
                    draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill="red")

            if save_path:
                img.save(save_path)
                print(f"Output image saved to {save_path}")
            
        except Exception as e:
            print(f"Could not plot results for task '{task}' with answer '{answer}'. Error: {e}")


    def simple_chat(self, prompt, image_path, enable_thinking=False, do_sample=True):
        """
        A simple chat function.
        """
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])
        
        response, history = self.model.chat(self.tokenizer, query=query, history=None, do_sample=do_sample)

        if enable_thinking and "<THINK>" in response and "</THINK>" in response:
            start_think = response.find("<THINK>") + len("<THINK>")
            end_think = response.find("</THINK>")
            thought = response[start_think:end_think].strip()
            answer_part = response[end_think + len("</THINK>"):].strip()
            # The executable part might be wrapped
            if "<EXECUTE>" in answer_part:
                 answer = answer_part.split("<EXECUTE>")[-1].strip()
            else:
                 answer = answer_part

        else:
            thought = ""
            if "<EXECUTE>" in response:
                 answer = response.split("<EXECUTE>")[-1].strip()
            else:
                answer = response.strip()
            
        return {"thinking": thought, "answer": answer}

if __name__ == '__main__':
    # Example usage:
    model = SimpleInference("BAAI/RoboBrain2.0-7B")
    
    prompt = "the person wearing a red hat"
    image = "./assets/demo/grounding.jpg"
    
    pred = model.inference(prompt, image, task="grounding", plot=True, save_path="output_grounding.jpg", enable_thinking=True)
    print(f"Prediction:\n{pred}")