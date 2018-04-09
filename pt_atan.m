function out = pt_atan(vector)
%Function returns the point-wise arctan of the input gector
	len = length(vector);
	out = zeros(1,len);
	for i=1:len
		out(i)=(atan(vector(i))/pi)+0.5;
	end
end