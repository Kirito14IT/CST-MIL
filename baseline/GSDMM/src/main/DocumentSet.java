package main;
import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import org.json.JSONObject;

public class DocumentSet{
	int D = 0;
	ArrayList<Document> documents = new ArrayList<Document>();

	public DocumentSet() {}

	public static DocumentSet fromPlain(String path, HashMap<String, Integer> vocabulary)
			throws Exception {
		DocumentSet result = new DocumentSet();
		try (BufferedReader in = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
			String line;
			while ((line = in.readLine()) != null) {
				result.documents.add(new Document(line, vocabulary, false));
				result.D++;
			}
		}
		return result;
	}
	
	public DocumentSet(String dataDir, HashMap<String, Integer> wordToIdMap) 
			 					throws Exception
	{
		BufferedReader in = Files.newBufferedReader(Paths.get(dataDir), StandardCharsets.UTF_8);
		String line;
		
		while((line=in.readLine()) != null){
			D++;
			JSONObject obj = new JSONObject(line);
			String text = obj.getString("text");
			Document document = new Document(text, wordToIdMap);
			documents.add(document);
		}
		
		in.close();
	}
}
