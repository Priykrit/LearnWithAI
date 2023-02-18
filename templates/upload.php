<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  if (isset($_FILES["file"]) && $_FILES["file"]["error"] == 0) {
    $file = $_FILES["file"];
    
    $uploadDir = "uploads/";
    
    $filename = uniqid() . "-" . basename($file["name"]);
    
    if (move_uploaded_file($file["tmp_name"], $uploadDir . $filename)) {
      echo "File uploaded successfully.";
    } else {
      echo "Error uploading file.";
    }
  } else {
    echo "No file was uploaded.";
  }
}
?>
